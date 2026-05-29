"""ALPR pipeline: YOLO plate detector + EasyOCR + Indian plate regex validation.

Default detector is the YOLOv8x-OIV7 engine (class #568 "Vehicle registration
plate") because it's already on disk and good enough for v1.  Swap in a
dedicated license-plate YOLO by passing a different `detector_path` — the rest
of the pipeline doesn't care, it just filters by class name containing "plate".
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Indian plate formats:
#   KA01AB1234       -- standard
#   KA01A1234        -- single-letter series
#   KL08AC1234       -- standard
#   DL3CAY0001       -- 1-letter district sometimes
#   22BH1234A        -- Bharat series: YY (year) + BH + 4 digits + 1 letter
#   00D012345        -- military / diplomatic — skipped here
INDIAN_PLATE_REGEXES = (
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$"),   # standard
    re.compile(r"^\d{2}BH\d{4}[A-Z]$"),                # BH-series
)

# Characters EasyOCR commonly confuses on Indian plates.  Applied carefully:
# we do NOT replace inside letter positions vs digit positions blindly, but a
# light normalisation helps with the obvious O/0, I/1, S/5 mistakes when
# regex-validating.  See `_clean_text` below.
_CHAR_FIXES = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}


@dataclass
class PlateRead:
    text: str           # the cleaned OCR text
    raw_text: str       # the un-cleaned OCR text
    det_conf: float     # detector confidence
    ocr_conf: float     # mean OCR confidence over the read
    bbox: tuple[int, int, int, int]   # x1,y1,x2,y2 in source-image coordinates
    valid: bool         # passes Indian plate regex?
    track_id: int | None = None


class ALPRPipeline:
    """One-shot ALPR over a single BGR frame.

    Construct once (loads detector + OCR backend, a few seconds), then call
    `detect_and_read(frame)` per frame.  Returns a list of `PlateRead`.

    OCR backends:
      - "fast"    -> fast-plate-ocr (ONNX, ~6 ms/plate, plate-specific training).
                     Default.  Loads `cct-xs-v1-global-model` (~2 MB).
      - "paddle"  -> PaddleOCR PP-OCR recognizer (~30 ms/plate on CPU, very accurate
                     general-purpose Chinese/English OCR with strong digit recall).
                     We use det=False since YOLO already gave us the crop.
      - "easyocr" -> EasyOCR (~50-100 ms/plate, general-purpose).  Fallback.
    """

    def __init__(self,
                 detector_path: str | Path = "/home/jetson/alpr/license-plate-finetune-v1l.engine",
                 detector_task: str = "detect",
                 imgsz: int = 640,
                 det_conf: float = 0.20,
                 plate_class_keywords: tuple[str, ...] = ("plate", "registration"),
                 ocr_backend: str = "fast",
                 fast_model: str = "global-plates-mobile-vit-v2-model",
                 ocr_langs: tuple[str, ...] = ("en",),
                 ocr_gpu: bool = True,
                 min_plate_area_px: int = 400,
                 ):
        self.detector = YOLO(str(detector_path), task=detector_task)
        self.imgsz = imgsz
        self.det_conf = det_conf
        self.plate_class_keywords = plate_class_keywords
        self.min_plate_area_px = min_plate_area_px
        self.ocr_backend = ocr_backend.lower()

        # Initialise lazily — only the chosen backend incurs its import cost.
        self._fast = None
        self._easyocr = None
        self._paddle = None
        self._paddle_api = None      # "v3" or "v2" — paddleocr's call signature changed

        if self.ocr_backend == "fast":
            # fast-plate-ocr renamed `LicensePlateRecognizer` → `ONNXPlateRecognizer`
            # in 0.3.0.  Try the new name first, fall back to the legacy class
            # so the same code base works against both API generations.
            try:
                from fast_plate_ocr import ONNXPlateRecognizer as _FastCls  # noqa: WPS433
            except ImportError:
                from fast_plate_ocr import LicensePlateRecognizer as _FastCls  # noqa: WPS433
            self._fast = _FastCls(fast_model)
        elif self.ocr_backend == "easyocr":
            import easyocr  # noqa: WPS433
            self._easyocr = easyocr.Reader(list(ocr_langs), gpu=ocr_gpu, verbose=False)
        elif self.ocr_backend == "paddle":
            self._init_paddle()
        else:
            raise ValueError(
                f"unknown ocr_backend {ocr_backend!r}; pick 'fast', 'paddle' or 'easyocr'"
            )

        # Cache the set of class indices we treat as "plate".
        names = self.detector.names
        self.plate_class_ids = {
            i for i, n in names.items()
            if any(kw.lower() in n.lower() for kw in plate_class_keywords)
        }
        if not self.plate_class_ids:
            print(f"WARN: detector has no class matching {plate_class_keywords!r}; "
                  f"falling back to ALL detections as plate candidates.")
        # half precision only meaningful for .engine
        self._is_engine = Path(detector_path).suffix == ".engine"

    # ---------------------------------------------------------------- detector

    def detect_plates(self, frame: np.ndarray):
        """Return a list of (bbox_xyxy_int, det_conf)."""
        res = self.detector.predict(
            frame, imgsz=self.imgsz, device=0, half=self._is_engine,
            conf=self.det_conf, verbose=False,
        )[0]
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cls = int(b.cls)
            if self.plate_class_ids and cls not in self.plate_class_ids:
                continue
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            if (x2 - x1) * (y2 - y1) < self.min_plate_area_px:
                continue
            out.append(((x1, y1, x2, y2), float(b.conf)))
        return out

    # --------------------------------------------------------------------- OCR

    @staticmethod
    def _clean_text(s: str) -> str:
        """Strip whitespace + punctuation, uppercase, keep [A-Z0-9] only.

        Indian plates carry an `IND` country-code watermark in the top-left
        corner, plus a hologram/serial number along the bottom edge.  Both
        get read alongside the registration text by good OCR (e.g. Florence-2).
        We drop:
          - leading `IND` (if the residue is plausibly a plate)
          - trailing `IND`
          - any `IND...` segment that appears AFTER a plausible plate prefix
            (e.g. `WB06F5977INDDA16006884` -> `WB06F5977`)
        """
        s = "".join(ch for ch in s.upper() if ch.isalnum())
        if s.startswith("IND") and len(s) >= 11:
            s = s[3:]
        # truncate at first mid-string IND (after position 6 — anything less
        # would be too short to be a real plate)
        idx = s.find("IND", 6)
        if idx > 0:
            s = s[:idx]
        if s.endswith("IND") and len(s) >= 11:
            s = s[:-3]
        return s

    def _ocr_crop(self, crop: np.ndarray) -> tuple[str, str, float]:
        """Dispatch to the selected OCR backend.

        Returns (clean_text, raw_text, ocr_confidence).
        """
        if crop.size == 0:
            return "", "", 0.0
        if self.ocr_backend == "fast":
            return self._ocr_fast(crop)
        if self.ocr_backend == "paddle":
            return self._ocr_paddle(crop)
        return self._ocr_easyocr(crop)

    # --------------------------------------------------------------- paddleocr

    def _init_paddle(self) -> None:
        """Construct a PaddleOCR recognizer, supporting both 2.x and 3.x APIs.

        We feed it pre-cropped plates from YOLO so we want **rec only** — no
        text-detection stage.  PaddleOCR v3 uses `.predict()` and a different
        kwarg set; v2 keeps the legacy `det=False, rec=True, cls=False` on
        `.ocr()`.  We dispatch by package version because 2.7+ silently
        accepts (and ignores) the v3 kwargs, so a TypeError fallback is
        unreliable.
        """
        import paddleocr  # noqa: WPS433
        from paddleocr import PaddleOCR  # noqa: WPS433
        try:
            major = int(str(paddleocr.__version__).split(".")[0])
        except (AttributeError, ValueError):
            major = 2
        if major >= 3:
            self._paddle = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="en",
            )
            self._paddle_api = "v3"
        else:
            self._paddle = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                show_log=False,
            )
            self._paddle_api = "v2"

    def _ocr_paddle(self, crop: np.ndarray) -> tuple[str, str, float]:
        """PaddleOCR recognizer (PP-OCRv4/v5 english).  ~20-30 ms on Orin CPU.

        Tiny crops get upscaled — PP-OCR's rec head was trained on >= 32 px tall
        text and stutters on smaller inputs.
        """
        h, w = crop.shape[:2]
        if max(h, w) < 200:
            scale = 200 / max(h, w)
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)
        try:
            if self._paddle_api == "v3":
                # v3: predict() returns a list of dicts with 'rec_texts' / 'rec_scores'
                preds = self._paddle.predict(crop)
                texts, confs = [], []
                for p in (preds or []):
                    d = p if isinstance(p, dict) else getattr(p, "json", lambda: {})()
                    rt = d.get("rec_texts") or d.get("rec_text") or []
                    rs = d.get("rec_scores") or d.get("rec_score") or []
                    if isinstance(rt, str):
                        rt = [rt]; rs = [rs] if not isinstance(rs, (list, tuple)) else rs
                    for t, s in zip(rt, rs):
                        if t:
                            texts.append(str(t)); confs.append(float(s or 0.0))
                if not texts:
                    return "", "", 0.0
                raw = " ".join(texts)
                clean = self._clean_text(raw)
                return clean, raw, float(np.mean(confs)) if confs else 0.0
            # v2 path: rec-only call
            res = self._paddle.ocr(crop, det=False, rec=True, cls=False)
        except Exception as e:    # noqa: BLE001
            print(f"WARN: paddleocr failed on a crop ({e})")
            return "", "", 0.0
        # v2 shape with det=False:  [[('TEXT', conf), ...]]  (sometimes [['TEXT', conf]])
        if not res or not res[0]:
            return "", "", 0.0
        rows = res[0]
        texts, confs = [], []
        for r in rows:
            if r is None:
                continue
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                texts.append(str(r[0]))
                try:
                    confs.append(float(r[1]))
                except (TypeError, ValueError):
                    pass
        if not texts:
            return "", "", 0.0
        raw = " ".join(texts)
        clean = self._clean_text(raw)
        return clean, raw, float(np.mean(confs)) if confs else 0.0

    def _ocr_fast(self, crop: np.ndarray) -> tuple[str, str, float]:
        """fast-plate-ocr: plate-specific, ~6 ms on CPU.

        The model handles its own internal resize, so we just hand it the
        full plate crop.  Returns clean=raw=cleaned-text and confidence=1.0
        (the library doesn't expose per-prediction confidence at this API).
        """
        preds = self._fast.run(crop)
        if not preds:
            return "", "", 0.0
        raw = preds[0].plate
        clean = self._clean_text(raw)
        # fast-plate-ocr's `char_probs` is None by default; treat any prediction
        # as confidence 1.0 (downstream voting+regex still applies).
        return clean, raw, 1.0

    def _ocr_easyocr(self, crop: np.ndarray) -> tuple[str, str, float]:
        """EasyOCR: general-purpose, ~50-100 ms.

        Downscales crops > 720 px wide because EasyOCR over-fires on
        background patterns (carbon-fibre etc.) when given too much resolution.
        """
        h, w = crop.shape[:2]
        TARGET_W = 480
        if w > TARGET_W * 1.5:
            scale = TARGET_W / w
            crop = cv2.resize(crop, (TARGET_W, int(h * scale)),
                              interpolation=cv2.INTER_AREA)
        elif max(h, w) < 200:
            scale = 200 / max(h, w)
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)
        results = self._easyocr.readtext(
            crop,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            paragraph=False,
            low_text=0.5,
        )
        if not results:
            return "", "", 0.0
        results.sort(key=lambda r: r[0][0][0])
        raw = " ".join(r[1] for r in results)
        confs = [r[2] for r in results]
        clean = self._clean_text(raw)
        return clean, raw, float(np.mean(confs)) if confs else 0.0

    # ----------------------------------------------------------- regex / fixes

    @staticmethod
    def _regex_valid(text: str) -> bool:
        return any(rx.match(text) for rx in INDIAN_PLATE_REGEXES)

    @classmethod
    def _try_char_fixes(cls, text: str) -> tuple[str, bool]:
        """Try a few O/0, I/1 style substitutions to coerce into a valid plate.
        Returns (best_text, valid).  Bounded search — at most 2 swaps.
        """
        if cls._regex_valid(text):
            return text, True
        # try simple swap of any one char
        for i, ch in enumerate(text):
            if ch in _CHAR_FIXES:
                alt = text[:i] + _CHAR_FIXES[ch] + text[i + 1:]
                if cls._regex_valid(alt):
                    return alt, True
        return text, False

    # ------------------------------------------------------------------ public

    def detect_and_read(self, frame: np.ndarray) -> list[PlateRead]:
        plates = self.detect_plates(frame)
        out: list[PlateRead] = []
        for (x1, y1, x2, y2), det_conf in plates:
            # pad crop a little — EasyOCR likes a bit of margin
            pad = 4
            ph, pw = frame.shape[:2]
            crop = frame[max(0, y1 - pad):min(ph, y2 + pad),
                         max(0, x1 - pad):min(pw, x2 + pad)]
            clean, raw, ocr_conf = self._ocr_crop(crop)
            fixed, valid = self._try_char_fixes(clean) if clean else (clean, False)
            out.append(PlateRead(
                text=fixed, raw_text=raw, det_conf=det_conf, ocr_conf=ocr_conf,
                bbox=(x1, y1, x2, y2), valid=valid,
            ))
        return out


# --------------------------------------------------------- quick self-test CLI

def _main():
    import argparse, sys
    p = argparse.ArgumentParser(description="ALPR self-test on a single image")
    p.add_argument("image", help="path to an image with vehicles/plates")
    p.add_argument("--model", default="/home/jetson/yolov8/yolov8x-oiv7.engine")
    p.add_argument("--conf", type=float, default=0.20)
    a = p.parse_args()
    frame = cv2.imread(a.image)
    if frame is None:
        sys.exit(f"could not read {a.image}")
    t0 = time.perf_counter()
    pipe = ALPRPipeline(detector_path=a.model, det_conf=a.conf)
    print(f"pipeline loaded in {time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter()
    reads = pipe.detect_and_read(frame)
    print(f"inference: {(time.perf_counter()-t0)*1000:.0f} ms")
    for r in reads:
        mark = "✓" if r.valid else "?"
        print(f"  {mark} {r.text or '(no text)':<14} "
              f"det={r.det_conf:.2f} ocr={r.ocr_conf:.2f} "
              f"raw={r.raw_text!r:<20} bbox={r.bbox}")
    # save annotated
    for r in reads:
        x1, y1, x2, y2 = r.bbox
        color = (0, 200, 0) if r.valid else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if r.text:
            cv2.putText(frame, r.text, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    out = Path(a.image).with_suffix(".alpr.jpg")
    cv2.imwrite(str(out), frame)
    print(f"annotated: {out}")


if __name__ == "__main__":
    _main()
