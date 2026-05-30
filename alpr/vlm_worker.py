"""Async VLM verification worker for the ALPR pipeline.

Runs a small vision-language model in a daemon thread.  Main thread submits
plate crops via `submit(track_id, crop_bgr)` and polls results via
`get(track_id)`.  The bounded queue drops oldest if main submits faster than
the VLM can process — intentional: live preview FPS must not drop.

Supported VLM backends:
- `microsoft/Florence-2-base`  (~230 M, ~500 MB FP16, ~0.5-1 s/plate)  ← default
- `microsoft/Florence-2-large` (~770 M, ~1.5 GB FP16, ~1-2 s/plate)
- `Qwen/Qwen2-VL-2B-Instruct`  (~2 B,  ~4 GB FP16, ~1.5-3 s/plate; tight on 8 GB Orin)

The backend is chosen by `model_id` prefix — `microsoft/Florence-2*` uses
Florence's `<OCR>` task mode; everything else assumes Qwen2-VL chat format.
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


# Qwen2-VL prompt (only used if you switch to a Qwen2-VL model)
QWEN_PROMPT = (
    "What is the license plate number visible in this image? "
    "Reply with only the alphanumeric characters of the plate, "
    "no spaces, no punctuation, no country code, no explanation."
)
FLORENCE_TASK = "<OCR>"

# Keep only [A-Z0-9] and strip leading IND watermark — same convention as
# core.ALPRPipeline._clean_text but applied here so the worker's output is
# already comparable to the fast-OCR text.
_NONALNUM = re.compile(r"[^A-Z0-9]")


def _clean(s: str) -> str:
    """Same cleaning convention as core.ALPRPipeline._clean_text — keep in sync."""
    s = _NONALNUM.sub("", s.upper())
    if s.startswith("IND") and len(s) >= 11:
        s = s[3:]
    # mid-string IND (plate followed by IND watermark/serial) — truncate at IND
    idx = s.find("IND", 6)
    if idx > 0:
        s = s[:idx]
    if s.endswith("IND") and len(s) >= 11:
        s = s[:-3]
    return s


@dataclass
class VLMResult:
    track_id: int
    text: str
    raw: str
    latency_s: float


class VLMWorker:
    """Background VLM verifier.  Florence-2-base by default.

    Usage:
        w = VLMWorker(); w.start()
        w.submit(track_id=3, crop=bgr_ndarray)
        text = w.get(3)           # None until ready
        ...
        w.stop()
    """

    def __init__(self,
                 model_id: str = "microsoft/Florence-2-base",
                 device: str = "cpu",
                 dtype: torch.dtype | None = None,
                 max_queue: int = 8,
                 max_new_tokens: int = 32):
        """
        Defaults to **CPU** because the 8 GB Orin Nano OOMs when loading
        a VLM on the GPU alongside the YOLO TRT engine + ONNX runtime + EasyOCR.
        Pass `device="cuda"` only on Orin NX 16 GB / AGX Orin (32/64 GB).
        FP16 on CPU is slow on aarch64; we automatically pick FP32 for CPU
        unless the caller forces a dtype.
        """
        self.model_id = model_id
        self.device = device
        # default precision: FP16 on GPU, FP32 on CPU (FP16 has weak cpu kernels on aarch64)
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.backend = "florence" if "florence" in model_id.lower() else "qwen"

        self._q: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=max_queue)
        self._results: dict[int, VLMResult] = {}
        self._submitted: set[int] = set()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ready = threading.Event()

        self.model = None
        self.processor = None

    # ----- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="VLMWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        try:
            self._q.put_nowait(None)   # wake the worker
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        return self._ready.wait(timeout=timeout)

    # ----- submit / get -------------------------------------------------------

    def submit(self, track_id: int, crop_bgr: np.ndarray) -> bool:
        """Enqueue a plate crop for verification.

        Returns True if accepted, False if (a) this track was already submitted
        or (b) the queue is full and oldest was dropped.
        """
        with self._lock:
            if track_id in self._submitted:
                return False
            self._submitted.add(track_id)
        try:
            self._q.put_nowait((track_id, crop_bgr))
            return True
        except queue.Full:
            # drop oldest, enqueue new
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait((track_id, crop_bgr))
            except queue.Full:
                return False
            return True

    def get(self, track_id: int) -> Optional[VLMResult]:
        with self._lock:
            return self._results.get(track_id)

    def pending_count(self) -> int:
        return self._q.qsize()

    # ----- worker -------------------------------------------------------------

    def _run(self) -> None:
        # Load model + processor inside the thread so the main thread doesn't
        # block on it.  Signal `_ready` when done.
        print(f"  [VLM] loading {self.model_id} on {self.device} ({self.dtype}) …", flush=True)
        t0 = time.perf_counter()
        if self.backend == "florence":
            # Workaround: transformers >= 4.46 unconditionally probes
            # `model._supports_sdpa` during `__init__` (before honouring any
            # `attn_implementation` kwarg).  Florence-2's community-maintained
            # `modeling_florence2.py` (loaded via `trust_remote_code=True`)
            # doesn't declare that attribute, so the probe raises
            # `AttributeError: 'Florence2ForConditionalGeneration' object has
            # no attribute '_supports_sdpa'` and the VLM worker thread dies
            # before serving any frame.  Pre-set the attribute on the
            # PreTrainedModel base class so all subclasses inherit a sane
            # default of False (which makes transformers fall back to eager
            # attention — the right choice for Florence-2 anyway).
            import transformers as _tfm   # noqa: WPS433
            if getattr(_tfm.PreTrainedModel, "_supports_sdpa", None) is None:
                _tfm.PreTrainedModel._supports_sdpa = False
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True,
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration  # noqa: WPS433
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model.eval()
        print(f"  [VLM] ready in {time.perf_counter()-t0:.1f}s", flush=True)
        self._ready.set()

        while not self._stop_event.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            track_id, crop_bgr = item
            try:
                result = self._infer(track_id, crop_bgr)
                with self._lock:
                    self._results[track_id] = result
                print(f"  [VLM] track {track_id}: {result.text!r}  "
                      f"({result.latency_s*1000:.0f} ms)", flush=True)
            except Exception as e:    # noqa: BLE001
                print(f"  [VLM] track {track_id} ERR: {e}", flush=True)
            finally:
                # Free Florence-2's vision-encoder activations + transient
                # decoder buffers between plates.  On the 8 GB Orin Nano
                # this is the difference between "VLM works for one plate
                # then OOMs on the next" and "VLM works for every plate".
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _infer(self, track_id: int, crop_bgr: np.ndarray) -> VLMResult:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        if self.backend == "florence":
            inputs = self.processor(text=FLORENCE_TASK, images=img,
                                    return_tensors="pt")
            inputs = {k: v.to(self.device, self.dtype if v.is_floating_point() else v.dtype)
                      for k, v in inputs.items()}
            t0 = time.perf_counter()
            with torch.inference_mode():
                # `use_cache=False` is critical here: transformers >= 4.46
                # switched past_key_values from the legacy tuple-of-tuples
                # format to a `DynamicCache` instance, but Florence-2's
                # community-maintained `modeling_florence2.py` still does
                # `past_key_values[0][0].shape[2]` in
                # `prepare_inputs_for_generation` (line 2197).  With the
                # new cache that lookup returns None and raises
                # `AttributeError: 'NoneType' object has no attribute
                # 'shape'`.  Disabling the cache sidesteps that whole
                # codepath — slightly slower per-token but Florence-2
                # generates < 64 tokens for plate OCR, so the impact is
                # negligible.
                # Use greedy decode (num_beams=1) on the Orin Nano 8 GB so
                # GPU mode fits.  Beam search with num_beams=3 needs ~2.5 GB
                # of vision-encoder + decoder workspace per beam, which
                # pushes total memory past what's available alongside the
                # YOLO TensorRT engine + PyTorch CUDA overhead.  Plate OCR
                # is unambiguous text so greedy is plenty.  Override with
                # VLM_NUM_BEAMS=N on larger Jetsons if you want beam search.
                out = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=self.max_new_tokens,
                    num_beams=int(os.environ.get("VLM_NUM_BEAMS", "1")),
                    do_sample=False,
                    use_cache=False,
                )
            raw_full = self.processor.batch_decode(out, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(
                raw_full, task=FLORENCE_TASK, image_size=(img.width, img.height),
            )
            raw = str(parsed.get(FLORENCE_TASK, "")).strip()
            latency = time.perf_counter() - t0
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": QWEN_PROMPT},
                ],
            }]
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text_prompt], images=[img], padding=True, return_tensors="pt",
            ).to(self.device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                )
            gen = out[:, inputs.input_ids.shape[1]:]
            raw = self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
            latency = time.perf_counter() - t0

        return VLMResult(
            track_id=track_id, text=_clean(raw), raw=raw, latency_s=latency,
        )
