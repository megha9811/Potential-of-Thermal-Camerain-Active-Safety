
# Integrated Rain with fog and dry, seperated 60mm and 20mm plots + results

from ultralytics import YOLO
from pathlib import Path
import numpy as np
import tifffile as tiff
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import sys


DATA_ROOT = Path("Data")

# If running dry or fog or rain
# "Measurements_Dry_03_11_2025" or "Fog_Measurements_28_11_2025" or "Rain_Measurements_22_12_2025"

DATASET_FOLDER = "Rain_Measurements_22_12_2025"
# Conditions to run fog or rain subfolders  for fog =  "Low_Visibility_Max_Fog_[F1-F59]", "Medium_Visibility_Fog_ramp_[F60-F79]"
# for rain = "Rain_20mm","Rain_60mm"

CONDITIONS = [
    "Rain_20mm", "Rain_60mm"
]

THERMAL_MODEL_PATH = "best.pt"
RGB_MODEL_PATH = "yolov8n.pt"

CONF_THRES = 0.5
IOU_THRES = 0.5

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
LOW_PCT, HIGH_PCT = 1.0, 99.0
EPS = 1e-6

# Default fallback thresholds
DEFAULT_REAL_THR = 0.52
DEFAULT_DUMMY_THR = 0.55

# Output base (will contain per-condition subfolders)
OUTPUT_BASE = DATA_ROOT / "output_images" / DATASET_FOLDER

# PROP PLACEMENT DISTANCES (meters) and prop name mapping

PROP_DISTANCES = {
    "cone": 5,
    "plant": 10,
    "suitcase": 15,
    "speed_limit_sign": 15,
    "bicycle_wheel": 20,
    "wheel": 20,
    "wooden_pallet": 25,
    "dead_pedestrian": 30,
    "cardboard_box": 35,
    "bicycle_with_dummy": 35,
    "backpack": 40,
    "chair": 40,
    "bicycle": 45,
    "person": 50,
}

# readable prop name mapping (for replacing "prop" in x-axis)
PROP_NAME_MAP = {
    "cone": "Cone",
    "plant": "Plant",
    "suitcase": "Suitcase",
    "speed_limit_sign": "Speed limit sign",
    "bicycle_wheel": "Bicycle wheel",
    "wheel": "Bicycle wheel",
    "cardboard_box": "Cardboard box",
    "dead_pedestrian": "Dead Pedestrian",
    "wooden_pallet": "Wooden pallet",
    "bicycle_with_dummy": "Dummy Cyclist",
    "backpack": "Backpack",
    "chair": "Chair",
    "bicycle": "Bicycle",
    "person": "Real / Dummy Pedestrian",
    "dummy_cyclist": "Dummy Cyclist",
    "real_pedestrian": "Real Pedestrian",
    "dummy_pedestrian": "Dummy Pedestrian",
}


# ZONE THRESHOLDS (M + F + R combined)

ZONE_THRESHOLDS = {
    "M1":  {"real": 0.52, "dummy": 0.55},
    "M2":  {"real": 0.52, "dummy": 0.55},
    "M3":  {"real": 0.52, "dummy": 0.55},
    "M5":  {"real": 0.52, "dummy": 0.55},
    "M7":  {"real": 0.52, "dummy": 0.55},
    "M8":  {"real": 0.52, "dummy": 0.55},
    "M9":  {"real": 0.52, "dummy": 0.55},
    "M11": {"real": 0.52, "dummy": 0.55},
    "M13": {"real": 0.52, "dummy": 0.55},
    "M14": {"real": 0.52, "dummy": 0.55},
    "M15": {"real": 0.52, "dummy": 0.55},
    "M17": {"real": 0.52, "dummy": 0.55},
    "M19": {"real": 0.52, "dummy": 0.55},
    "M20": {"real": 0.52, "dummy": 0.585},
    "M21": {"real": 0.52, "dummy": 0.585},
    "M23": {"real": 0.52, "dummy": 0.585},
    "M25": {"real": 0.5,  "dummy": 0.585},
    "M26": {"real": 0.5,  "dummy": 0.585},
    "M27": {"real": 0.5,  "dummy": 0.585},
    "M29": {"real": 0.5,  "dummy": 0.585},
    "M31": {"real": 0.52, "dummy": 0.585},
    "M32": {"real": 0.52, "dummy": 0.585},
    "M33": {"real": 0.52, "dummy": 0.585},
    "M35": {"real": 0.52, "dummy": 0.585},

}
# M37..M59 dummy=None
for m in ["M37", "M38", "M39", "M41", "M43", "M44", "M45", "M47", "M49", "M50", "M51", "M53", "M55", "M56", "M57", "M59"]:
    ZONE_THRESHOLDS[m] = {"real": 0.52, "dummy": None}

# F thresholds (F1..F59)
F_thresholds = {
    "F1": {"real": 0.52, "dummy": 0.55},
    "F2": {"real": 0.52, "dummy": 0.55},
    "F3": {"real": 0.52, "dummy": 0.55},
    "F5": {"real": 0.52, "dummy": 0.55},
    "F7": {"real": 0.52, "dummy": 0.55},
    "F8": {"real": 0.52, "dummy": 0.55},
    "F9": {"real": 0.52, "dummy": 0.55},
    "F11": {"real": 0.52, "dummy": 0.55},
    "F13": {"real": 0.52, "dummy": 0.55},
    "F14": {"real": 0.52, "dummy": 0.55},
    "F15": {"real": 0.52, "dummy": 0.55},
    "F17": {"real": 0.52, "dummy": 0.55},
    "F19": {"real": 0.52, "dummy": 0.55},
    "F20": {"real": 0.52, "dummy": 0.585},
    "F21": {"real": 0.52, "dummy": 0.585},
    "F23": {"real": 0.52, "dummy": 0.585},
    "F25": {"real": 0.5, "dummy": 0.585},
    "F26": {"real": 0.5, "dummy": 0.585},
    "F27": {"real": 0.5, "dummy": 0.585},
    "F29": {"real": 0.5, "dummy": 0.585},
    "F31": {"real": 0.52, "dummy": 0.585},
    "F32": {"real": 0.52, "dummy": 0.585},
    "F33": {"real": 0.52, "dummy": 0.585},
    "F35": {"real": 0.52, "dummy": 0.585},
}
for f in ["F37", "F38", "F39", "F41", "F43", "F44", "F45", "F47", "F49", "F50", "F51", "F53", "F55", "F56", "F57", "F59"]:
    F_thresholds[f] = {"real": 0.52, "dummy": None}
for k, v in F_thresholds.items():
    ZONE_THRESHOLDS[k] = v.copy()


# F60..F74 thresholds for Detections but excluded from plots)

F_thresholds_extra = {
    "F60": {"real": 0.52, "dummy": None},
    "F61": {"real": 0.52, "dummy": None},
    "F62": {"real": 0.52, "dummy": 0.585},
    "F63": {"real": 0.52, "dummy": 0.585},
    "F64": {"real": 0.52, "dummy": 0.585},
    "F70": {"real": 0.52, "dummy": 0.55},
    "F71": {"real": 0.52, "dummy": 0.55},
    "F72": {"real": 0.52, "dummy": 0.55},
    "F73": {"real": 0.52, "dummy": 0.55},
    "F74": {"real": 0.52, "dummy": 0.585},
}
for k, v in F_thresholds_extra.items():
    ZONE_THRESHOLDS[k] = v.copy()

RAIN_TEST_NUMBERS = [
    1, 2, 3, 5, 7, 8, 9, 11, 13, 14, 15, 17, 19, 20, 21, 23,
    25, 26, 27, 29, 31, 32, 33, 35, 37, 38, 39, 41,
    43, 44, 45, 47, 49, 50, 51, 53, 55, 56, 57, 59
]


def rain_thresholds(num):
    if num <= 23:
        return 0.52, 0.55
    if num in (25, 26, 27, 29):
        return 0.53, 0.585
    if num in (31, 32, 33, 35):
        return 0.52, 0.585
    return 0.52, None


# Rain thresholds
for rain_prefix in ("R20", "R60"):
    for n in RAIN_TEST_NUMBERS:
        real_thr, dummy_thr = rain_thresholds(n)
        ZONE_THRESHOLDS[f"{rain_prefix}_{n}"] = {
            "real": real_thr,
            "dummy": dummy_thr
        }


# TEST CASE INFO (vehicle positions & lights)
# note: F60..F79 excluded
# distances are VEHICLE position from start (0,5,10,...)

TEST_CASE_INFO = {
    # M series: vehicle position from start (0,5,..45)
    "M1":  {"distance": 0,  "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "M2":  {"distance": 0,  "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "M3":  {"distance": 0,  "lab": "ON(2%)",   "car": "Low beam"},
    "M5":  {"distance": 0,  "lab": "OFF",      "car": "Low beam"},
    "M7":  {"distance": 5,  "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "M8":  {"distance": 5,  "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "M9":  {"distance": 5,  "lab": "ON(2%)",   "car": "Low beam"},
    "M11": {"distance": 5,  "lab": "OFF",      "car": "Low beam"},
    "M13": {"distance": 10, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "M14": {"distance": 10, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "M15": {"distance": 10, "lab": "ON(2%)",   "car": "Low beam"},
    "M17": {"distance": 10, "lab": "OFF",      "car": "Low beam"},
    "M19": {"distance": 15, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "M20": {"distance": 15, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "M21": {"distance": 15, "lab": "ON(2%)",   "car": "Low beam"},
    "M23": {"distance": 15, "lab": "OFF",      "car": "Low beam"},
    "M25": {"distance": 20, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "M26": {"distance": 20, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "M27": {"distance": 20, "lab": "ON(2%)",   "car": "Low beam"},
    "M29": {"distance": 20, "lab": "OFF",      "car": "Low beam"},
    "M31": {"distance": 25, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "M32": {"distance": 25, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "M33": {"distance": 25, "lab": "ON(2%)",   "car": "Low beam"},
    "M35": {"distance": 25, "lab": "OFF",      "car": "Low beam"},
    "M37": {"distance": 30, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "M38": {"distance": 30, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "M39": {"distance": 30, "lab": "ON(2%)",   "car": "Low beam"},
    "M41": {"distance": 30, "lab": "OFF",      "car": "Low beam"},
    "M43": {"distance": 35, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "M44": {"distance": 35, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "M45": {"distance": 35, "lab": "ON(2%)",   "car": "Low beam"},
    "M47": {"distance": 35, "lab": "OFF",      "car": "Low beam"},
    "M49": {"distance": 40, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "M50": {"distance": 40, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "M51": {"distance": 40, "lab": "ON(2%)",   "car": "Low beam"},
    "M53": {"distance": 40, "lab": "OFF",      "car": "Low beam"},
    "M55": {"distance": 45, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "M56": {"distance": 45, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "M57": {"distance": 45, "lab": "ON(2%)",   "car": "Low beam"},
    "M59": {"distance": 45, "lab": "OFF",      "car": "Low beam"},
}
# F1..F59 entries (F60..F79 excluded from plotting)
TEST_CASE_INFO.update({
    "F1":  {"distance": 0,  "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F2":  {"distance": 0,  "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "F3":  {"distance": 0,  "lab": "ON(2%)",   "car": "Low beam"},
    "F5":  {"distance": 0,  "lab": "OFF",      "car": "Low beam"},
    "F7":  {"distance": 5,  "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F8":  {"distance": 5,  "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "F9":  {"distance": 5,  "lab": "ON(2%)",   "car": "Low beam"},
    "F11": {"distance": 5,  "lab": "OFF",      "car": "Low beam"},
    "F13": {"distance": 10, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F14": {"distance": 10, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "F15": {"distance": 10, "lab": "ON(2%)",   "car": "Low beam"},
    "F17": {"distance": 10, "lab": "OFF",      "car": "Low beam"},
    "F19": {"distance": 15, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F20": {"distance": 15, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "F21": {"distance": 15, "lab": "ON(2%)",   "car": "Low beam"},
    "F23": {"distance": 15, "lab": "OFF",      "car": "Low beam"},
    "F25": {"distance": 20, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F26": {"distance": 20, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "F27": {"distance": 20, "lab": "ON(2%)",   "car": "Low beam"},
    "F29": {"distance": 20, "lab": "OFF",      "car": "Low beam"},
    "F31": {"distance": 25, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F32": {"distance": 25, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "F33": {"distance": 25, "lab": "ON(2%)",   "car": "Low beam"},
    "F35": {"distance": 25, "lab": "OFF",      "car": "Low beam"},
    "F37": {"distance": 30, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F38": {"distance": 30, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "F39": {"distance": 30, "lab": "ON(2%)",   "car": "Low beam"},
    "F41": {"distance": 30, "lab": "OFF",      "car": "Low beam"},
    "F43": {"distance": 35, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F44": {"distance": 35, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "F45": {"distance": 35, "lab": "ON(2%)",   "car": "Low beam"},
    "F47": {"distance": 35, "lab": "OFF",      "car": "Low beam"},
    "F49": {"distance": 40, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F50": {"distance": 40, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "F51": {"distance": 40, "lab": "ON(2%)",   "car": "Low beam"},
    "F53": {"distance": 40, "lab": "OFF",      "car": "Low beam"},
    "F55": {"distance": 45, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F56": {"distance": 45, "lab": "ON(2%)",   "car": "Off (Day driving lights)"},
    "F57": {"distance": 45, "lab": "ON(2%)",   "car": "Low beam"},
    "F59": {"distance": 45, "lab": "OFF",      "car": "Low beam"},
})

# Included F60..F74 tests for Detection (but exclude from plots)

TEST_CASE_INFO.update({
    "F60": {"distance": 45, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F61": {"distance": 40, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F62": {"distance": 35, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F63": {"distance": 30, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F64": {"distance": 25, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F70": {"distance": 0,  "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F71": {"distance": 5,  "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F72": {"distance": 10, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F73": {"distance": 15, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
    "F74": {"distance": 20, "lab": "ON(100%)", "car": "Off (Day driving lights)"},
})

# Rain Test cases
RAIN_CASE_TEMPLATE = {
    1:  (0,  "ON(100%)", "Off (Day driving lights)"),
    2:  (0,  "ON(2%)",   "Off (Day driving lights)"),
    3:  (0,  "ON(2%)",   "Low beam"),
    5:  (0,  "OFF",      "Low beam"),
    7:  (5,  "ON(100%)", "Off (Day driving lights)"),
    8:  (5,  "ON(2%)",   "Off (Day driving lights)"),
    9:  (5,  "ON(2%)",   "Low beam"),
    11: (5,  "OFF",      "Low beam"),
    13: (10, "ON(100%)", "Off (Day driving lights)"),
    14: (10, "ON(2%)",   "Off (Day driving lights)"),
    15: (10, "ON(2%)",   "Low beam"),
    17: (10, "OFF",      "Low beam"),
    19: (15, "ON(100%)", "Off (Day driving lights)"),
    20: (15, "ON(2%)",   "Off (Day driving lights)"),
    21: (15, "ON(2%)",   "Low beam"),
    23: (15, "OFF",      "Low beam"),
    25: (20, "ON(100%)", "Off (Day driving lights)"),
    26: (20, "ON(2%)",   "Off (Day driving lights)"),
    27: (20, "ON(2%)",   "Low beam"),
    29: (20, "OFF",      "Low beam"),
    31: (25, "ON(100%)", "Off (Day driving lights)"),
    32: (25, "ON(2%)",   "Off (Day driving lights)"),
    33: (25, "ON(2%)",   "Low beam"),
    35: (25, "OFF",      "Low beam"),
    37: (30, "ON(100%)", "Off (Day driving lights)"),
    38: (30, "ON(2%)",   "Off (Day driving lights)"),
    39: (30, "ON(2%)",   "Low beam"),
    41: (30, "OFF",      "Low beam"),
    43: (35, "ON(100%)", "Off (Day driving lights)"),
    44: (35, "ON(2%)",   "Off (Day driving lights)"),
    45: (35, "ON(2%)",   "Low beam"),
    47: (35, "OFF",      "Low beam"),
    49: (40, "ON(100%)", "Off (Day driving lights)"),
    50: (40, "ON(2%)",   "Off (Day driving lights)"),
    51: (40, "ON(2%)",   "Low beam"),
    53: (40, "OFF",      "Low beam"),
    55: (45, "ON(100%)", "Off (Day driving lights)"),
    56: (45, "ON(2%)",   "Off (Day driving lights)"),
    57: (45, "ON(2%)",   "Low beam"),
    59: (45, "OFF",      "Low beam"),
}

for rain_prefix in ("R20", "R60"):
    for n, (dist, lab, car) in RAIN_CASE_TEMPLATE.items():
        TEST_CASE_INFO[f"{rain_prefix}_{n}"] = {
            "distance": dist,
            "lab": lab,
            "car": car,
        }


# Utilities: TIFF reading + Prepare_for_yolo


def read_image_any(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        try:
            arr = tiff.imread(str(path))
        except Exception:
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise RuntimeError(
                    f"Failed to read TIFF (tifffile & cv2): {path}")
            return arr
        if arr is None:
            raise RuntimeError(f"tifffile returned None for {path}")
        if not isinstance(arr, np.ndarray):
            raise RuntimeError(
                f"tifffile returned unexpected type {type(arr)} for {path}")
        if arr.ndim >= 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[2]:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]
        if arr.ndim > 3 or arr.ndim == 0 or arr.size == 0:
            raise RuntimeError(
                f"Unsupported TIFF shape {getattr(arr,'shape',None)} for {path}")
        return arr
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cv2 failed to read image: {path}")
    return img


def percentile_stretch_to_u8(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        raise ValueError("percentile_stretch_to_u8 got None")
    arr_f = arr.astype(np.float32)
    if arr_f.ndim == 3 and arr_f.shape[0] in (1, 3, 4) and arr_f.shape[0] != arr_f.shape[2]:
        arr_f = np.moveaxis(arr_f, 0, -1)
    if arr_f.ndim > 3:
        arr_f = arr_f.reshape((-1,) + arr_f.shape[-2:])[0]
    if arr_f.ndim == 3 and arr_f.shape[2] == 1:
        arr_f = arr_f[..., 0]
    if arr_f.ndim == 2:
        lo, hi = np.percentile(arr_f, (LOW_PCT, HIGH_PCT))
        scale = 255.0 / max(hi - lo, EPS)
        return np.clip((arr_f - lo) * scale, 0, 255).astype(np.uint8)
    if arr_f.ndim == 3:
        out = np.empty(arr_f.shape, dtype=np.uint8)
        for c in range(arr_f.shape[2]):
            ch = arr_f[..., c]
            lo, hi = np.percentile(ch, (LOW_PCT, HIGH_PCT))
            scale = 255.0 / max(hi - lo, EPS)
            out[..., c] = np.clip((ch - lo) * scale, 0, 255).astype(np.uint8)
        return out
    raise ValueError(
        f"Unsupported shape for percentile stretch: ndim={arr_f.ndim}, shape={arr_f.shape}")


def ensure_3ch_u8(img_u8: np.ndarray) -> np.ndarray:
    if img_u8.ndim == 2:
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    if img_u8.ndim == 3:
        c = img_u8.shape[2]
        if c == 1:
            return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        if c == 3:
            return img_u8
        if c == 4:
            return cv2.cvtColor(img_u8, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def prepare_for_yolo(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Failed to read image.")
    src = img
    if src.ndim == 3 and src.shape[2] == 1:
        src = src[..., 0]
    if src.dtype == np.uint16 or src.dtype == np.uint32 or np.issubdtype(src.dtype, np.floating) or src.dtype == np.int16:
        img_u8 = percentile_stretch_to_u8(src.astype(np.float32))
    elif src.dtype != np.uint8:
        img_u8 = percentile_stretch_to_u8(src.astype(np.float32))
    else:
        img_u8 = src
    return ensure_3ch_u8(img_u8)


# Helpers for rows, drawing, plotting

def normalize_test_case(tc: str) -> str:
    tc = tc.strip()
    if not tc:
        return tc
    if tc[0].upper() not in ("M", "F"):
        return tc
    try:
        num = int(tc[1:])
        return f"{tc[0].upper()}{num}"
    except Exception:
        return tc


def safe_class_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def readable_label_from_class(class_name: str) -> str:
    if not class_name:
        return "Object"
    return " ".join([p.capitalize() for p in class_name.split("_")])


def prop_distance_for_class(class_name: str, pedestrian_type: str = ""):
    name = (class_name or "").lower()
    if name == "person":
        if pedestrian_type == "dummy_cyclist":
            return PROP_DISTANCES.get("bicycle_with_dummy")
        return PROP_DISTANCES.get("person")
    if name in PROP_DISTANCES:
        return PROP_DISTANCES[name]
    for key in PROP_DISTANCES:
        if key in name:
            return PROP_DISTANCES[key]
    return None


def prop_name_for_plot(class_name: str, pedestrian_type: str = "") -> str:
    if class_name == "person":
        if pedestrian_type == "dummy_cyclist":
            return PROP_NAME_MAP.get("dummy_cyclist")
        if pedestrian_type == "real_pedestrian":
            return PROP_NAME_MAP.get("real_pedestrian")
        if pedestrian_type == "dummy_pedestrian":
            return PROP_NAME_MAP.get("dummy_pedestrian")
        return PROP_NAME_MAP.get("person")
    name = (class_name or "").lower()
    for key in PROP_NAME_MAP:
        if key in name:
            return PROP_NAME_MAP[key]
    return readable_label_from_class(class_name)


def detections_to_rows(result, image_path: Path, domain: str, test_case_folder: str, image_shape=None):
    rows = []
    base_tc = test_case_folder
    if base_tc.lower().endswith(" raw"):
        base_tc = base_tc[:-4].strip()
    tc_norm = normalize_test_case(base_tc)
    thr_info = ZONE_THRESHOLDS.get(tc_norm, None)
    if thr_info is None:
        real_thr = DEFAULT_REAL_THR
        dummy_thr = DEFAULT_DUMMY_THR
    else:
        real_thr = thr_info.get("real", DEFAULT_REAL_THR)
        dummy_thr = thr_info.get("dummy", DEFAULT_DUMMY_THR)
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
    else:
        h, w = result.orig_shape
    names = result.names
    if result.boxes is None or len(result.boxes) == 0:
        return rows
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    for i in range(len(cls)):
        x1, y1, x2, y2 = xyxy[i]
        class_id = int(cls[i])
        class_name = names.get(class_id, str(class_id))
        cx_norm = ((x1 + x2) / 2.0) / float(max(w, 1))
        pedestrian_type = ""
        if class_name == "person":
            if cx_norm < real_thr:
                pedestrian_type = "real_pedestrian"
            else:
                if dummy_thr is None:
                    pedestrian_type = "dummy_pedestrian"
                else:
                    if cx_norm < dummy_thr:
                        pedestrian_type = "dummy_pedestrian"
                    else:
                        pedestrian_type = "dummy_cyclist"
        prop_dist = prop_distance_for_class(class_name, pedestrian_type)
        rows.append({
            "domain":          domain,
            "test_case":       test_case_folder,
            "image_name":      image_path.name,
            "image_path":      str(image_path),
            "width":           int(w),
            "height":          int(h),
            "class_id":        class_id,
            "class_name":      class_name,
            "confidence":      float(conf[i]),
            "xmin":            float(x1),
            "ymin":            float(y1),
            "xmax":            float(x2),
            "ymax":            float(y2),
            "pedestrian_type": pedestrian_type,
            "prop_distance":   prop_dist,
        })
    return rows


def draw_detections_opencv(img: np.ndarray, rows: list):
    out = img.copy()
    h_img, w_img = out.shape[:2]
    for r in rows:
        x1 = int(max(0, min(w_img - 1, r["xmin"])))
        y1 = int(max(0, min(h_img - 1, r["ymin"])))
        x2 = int(max(0, min(w_img - 1, r["xmax"])))
        y2 = int(max(0, min(h_img - 1, r["ymax"])))
        conf = r.get("confidence", 0.0)
        cls = r.get("class_name", "")
        ped_type = r.get("pedestrian_type", "")
        if cls == "person":
            if ped_type == "real_pedestrian":
                label = "Real Pedestrian"
                color = (0, 200, 0)
            elif ped_type == "dummy_cyclist":
                label = "Dummy Cyclist"
                color = (255, 80, 80)
            elif ped_type == "dummy_pedestrian":
                label = "Dummy Pedestrian"
                color = (0, 120, 255)
            else:
                label = "Person"
                color = (0, 255, 255)
        else:
            label = readable_label_from_class(cls)
            color = (200, 200, 0)
        label_full = f"{label} {conf:.2f}"
        thickness = 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        (txt_w, txt_h), baseline = cv2.getTextSize(
            label_full, font, font_scale, font_thickness)
        pad = 6
        rect_x1 = x1
        rect_y1 = max(0, y1 - txt_h - baseline - pad)
        rect_x2 = min(w_img - 1, x1 + txt_w + pad)
        rect_y2 = y1
        if rect_x2 >= w_img:
            shift = rect_x2 - (w_img - 1)
            rect_x1 = max(0, rect_x1 - shift)
            rect_x2 = rect_x1 + txt_w + pad
        cv2.rectangle(out, (rect_x1, rect_y1),
                      (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.putText(out, label_full, (rect_x1 + 4, rect_y2 - baseline - 2),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return out


# Plotting relative distance & percent efficiency

# Appended Plot title formatting helpers + plot-exclude list-F60..F74 not used in plots

PLOT_EXCLUDE_TESTCASES = {"F60", "F61", "F62",
                          "F63", "F64", "F70", "F71", "F72", "F73", "F74"}


def dataset_tag_from_folder(dataset_folder: str) -> str:
    s = (dataset_folder or "").lower()
    if "dry" in s:
        return "Dry"
    if "fog" in s:
        return "Fog"
    return "Dataset"


def source_tag(src: str) -> str:
    if src == "8bit":
        return "8 Bit"
    if src == "16bit_norm":
        return "16 Bit→8 Bit"
    return str(src).replace("_", " ").title()


def condition_tag(condition_name: str) -> str:
    if condition_name is None:
        return ""
    s = condition_name.lower()
    if "rain_20mm" in s:
        return "Rain-20mm"
    if "rain_60mm" in s:
        return "Rain-60mm"
    return condition_name.replace("_", "-")


def format_plot_title(metric, dataset_folder, domain, src, prop_name, condition_name=None):
    prefix = condition_tag(
        condition_name) or dataset_tag_from_folder(dataset_folder)
    return (
        f"{prefix}-{domain}-{source_tag(src)}-{prop_name}: "
        f"{metric} vs Relative Distance"
    )


def choose_best_legend_loc(ax):
    # Choose a location that least overlaps curves (keeps legend INSIDE plot)
    # Matplotlib supports loc='best' for automatic placement.
    return "best"


def plot_distance_vs_conf_per_class(summary_df: pd.DataFrame, condition_name: str = None, plot_dir: Path = None):
    if summary_df.empty:
        return
    if "source_type" not in summary_df.columns:
        return

    # excluded F60..F74 from plotting but they remain in CSVs/detection
    if "test_case_norm" in summary_df.columns:
        summary_df = summary_df[~summary_df["test_case_norm"].isin(
            PLOT_EXCLUDE_TESTCASES)].copy()

    plot_dir = plot_dir or (OUTPUT_BASE / "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    cond_tag = safe_class_filename(condition_tag(
        condition_name)) + "_" if condition_name else ""
    for domain in sorted(summary_df["domain"].unique()):
        df_dom = summary_df[summary_df["domain"] == domain]
        for src in sorted(df_dom["source_type"].unique()):
            df_src = df_dom[df_dom["source_type"] == src]
            src_label = "8-bit camera images" if src == "8bit" else (
                "16-bit → 8-bit normalized" if src == "16bit_norm" else src)
            for cls_name in sorted(df_src["class_name"].unique()):
                sub = df_src[df_src["class_name"] == cls_name].copy()
                sub = sub.dropna(subset=["relative_distance"])
                if sub.empty:
                    continue
                first_row = sub.iloc[0]
                prop_name = prop_name_for_plot(
                    cls_name, first_row.get("pedestrian_type", ""))
                x_vals = sorted(
                    {int(round(x)) for x in sub["relative_distance"].dropna().astype(float).tolist()})
                fig, ax = plt.subplots(figsize=(9, 6))
                conditions = sorted(sub["condition"].dropna().unique())
                for cond in conditions:
                    cdf = sub[sub["condition"] == cond].copy()
                    if cdf.empty:
                        continue
                    cdf = cdf.sort_values("relative_distance")
                    ax.plot(cdf["relative_distance"],
                            cdf["mean_confidence"], marker="o", label=cond)
                ax.set_ylim(0, 1)
                ax.set_yticks([i/10 for i in range(0, 11)])
                x_max = float(sub["relative_distance"].max())
                ax.set_xlim(0, max(1, x_max))
                if x_vals:
                    ax.set_xticks(x_vals)

                ax.set_xlabel(f"Relative Distance Vehicle → {prop_name} [m]")
                ax.set_ylabel("Mean Detection Confidence [-]")

                ax.set_title(format_plot_title(
                    metric="Confidence",
                    dataset_folder=DATASET_FOLDER,
                    domain=domain,
                    src=src,
                    prop_name=prop_name,
                    condition_name=condition_name
                ))

                ax.grid(True, linestyle=":")

                # legend inside plot, auto placed
                ax.legend(title="Lab / Car lights", fontsize=8,
                          loc=choose_best_legend_loc(ax))

                fig.tight_layout()
                out_name = f"{cond_tag}{domain}_{src}_{safe_class_filename(cls_name)}_conf_vs_rel_distance.png"
                fig.savefig(plot_dir / out_name, dpi=180)
                plt.close(fig)


def plot_efficiency_per_class(summary_df: pd.DataFrame, condition_name: str = None, plot_dir: Path = None):
    if summary_df.empty or "efficiency" not in summary_df.columns:
        return
    if "source_type" not in summary_df.columns:
        return

    if "test_case_norm" in summary_df.columns:
        summary_df = summary_df[~summary_df["test_case_norm"].isin(
            PLOT_EXCLUDE_TESTCASES)].copy()

    plot_dir = plot_dir or (OUTPUT_BASE / "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    cond_tag = safe_class_filename(condition_tag(
        condition_name)) + "_" if condition_name else ""
    for domain in sorted(summary_df["domain"].unique()):
        df_dom = summary_df[summary_df["domain"] == domain]
        for src in sorted(df_dom["source_type"].unique()):
            df_src = df_dom[df_dom["source_type"] == src]
            src_label = "8-bit camera images" if src == "8bit" else (
                "16-bit → 8-bit normalized" if src == "16bit_norm" else src)
            for cls_name in sorted(df_src["class_name"].unique()):
                sub = df_src[df_src["class_name"] == cls_name].copy()
                sub = sub.dropna(subset=["relative_distance"])
                if sub.empty:
                    continue
                first_row = sub.iloc[0]
                prop_name = prop_name_for_plot(
                    cls_name, first_row.get("pedestrian_type", ""))
                x_vals = sorted(
                    {int(round(x)) for x in sub["relative_distance"].dropna().astype(float).tolist()})
                fig, ax = plt.subplots(figsize=(9, 6))
                conditions = sorted(sub["condition"].dropna().unique())
                for cond in conditions:
                    cdf = sub[sub["condition"] == cond].copy()
                    if cdf.empty:
                        continue
                    cdf = cdf.sort_values("relative_distance")
                    ax.plot(cdf["relative_distance"], cdf["efficiency"]
                            * 100.0, marker="o", label=cond)
                ax.set_ylim(0, 100)
                ax.set_yticks(list(range(0, 101, 10)))
                x_max = float(sub["relative_distance"].max())
                ax.set_xlim(0, max(1, x_max))
                if x_vals:
                    ax.set_xticks(x_vals)

                ax.set_xlabel(f"Relative Distance Vehicle → {prop_name} [m]")
                ax.set_ylabel("Detection Efficiency [%]")

                ax.set_title(format_plot_title(
                    metric="Efficiency",
                    dataset_folder=DATASET_FOLDER,
                    domain=domain,
                    src=src,
                    prop_name=prop_name,
                    condition_name=condition_name
                ))

                ax.grid(True, linestyle=":")

                ax.legend(title="Lab / Car lights", fontsize=8,
                          loc=choose_best_legend_loc(ax))

                fig.tight_layout()
                out_name = f"{cond_tag}{domain}_{src}_{safe_class_filename(cls_name)}_eff_vs_rel_distance_percent.png"
                fig.savefig(plot_dir / out_name, dpi=180)
                plt.close(fig)

# core processing for one domain Thermal or RGB
# writes per test case CSVs- existing
# Also writes domain level summary CSV: summary_{domain}_per_testcase_class.csv in out_root


def process_domain(domain_name: str, in_root: Path, out_root: Path, model: YOLO, summary_rows: list):
    if not in_root.exists():
        print(f"[WARN] input folder missing: {in_root}")
        return
    print(f"[INFO] Processing {domain_name} images under {in_root}")
    test_case_folders = [p for p in in_root.iterdir() if p.is_dir()]
    if not test_case_folders:
        print(f"[WARN] No test case folders found in {in_root}")
        return

    # domain_summary_rows collects the same dictionary rows appended to summary_rows
    domain_summary_rows = []

    for tc_folder in sorted(test_case_folders, key=lambda p: p.name):
        tc_name = tc_folder.name
        source_type = "8bit"
        base_tc_name = tc_name
        if tc_name.lower().endswith(" raw"):
            source_type = "16bit_norm"
            base_tc_name = tc_name[:-4].strip()
        out_tc_folder = out_root / tc_name
        out_tc_folder.mkdir(parents=True, exist_ok=True)
        all_rows = []
        images = [p for p in tc_folder.iterdir() if p.is_file()
                  and p.suffix.lower() in IMG_EXTS]
        if not images:
            print(f"[WARN]  No images in {tc_folder}")
            continue
        for img_path in sorted(images):
            try:
                img_raw = read_image_any(img_path)
                img_for_yolo = prepare_for_yolo(img_raw)
                results = model.predict(
                    source=img_for_yolo, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
                res = results[0]
                rows = detections_to_rows(
                    res, img_path, domain_name, tc_name, image_shape=img_for_yolo.shape[:2])
                all_rows.extend(rows)
                det_img = draw_detections_opencv(img_for_yolo, rows)
                det_name = img_path.stem + "_det.png"
                det_path = out_tc_folder / det_name
                cv2.imwrite(str(det_path), det_img)
            except Exception as e:
                print(f"[WARN] processing image {img_path}: {e}")
                continue

        # per test case CSV
        csv_path = out_tc_folder / f"{tc_name}_detections.csv"
        df_tc = pd.DataFrame(all_rows)
        if df_tc.empty:
            df_tc = pd.DataFrame(columns=[
                "domain", "test_case", "image_name", "image_path",
                "width", "height", "class_id", "class_name",
                "confidence", "xmin", "ymin", "xmax", "ymax",
                "pedestrian_type", "prop_distance",
            ])
        df_tc.to_csv(csv_path, index=False)
        print(f"[INFO]  Wrote CSV for {tc_name}: {csv_path}")

        if not all_rows:
            continue

        # compute class level mean confidence for the test case
        df_classes = (
            df_tc
            .groupby(["class_name", "pedestrian_type", "prop_distance"], dropna=False)["confidence"]
            .mean()
            .reset_index()
        )

        tc_norm = normalize_test_case(base_tc_name)
        info = TEST_CASE_INFO.get(tc_norm, {})
        vehicle_pos = info.get("distance")
        lab = info.get("lab")
        car = info.get("car")
        condition = f"Lab: {lab}, Car: {car}" if lab is not None and car is not None else None
        img_total = len(images)

        for _, r in df_classes.iterrows():
            cls_name = r["class_name"]
            ped_type = r["pedestrian_type"]
            prop_dist = r["prop_distance"]
            if prop_dist is None or vehicle_pos is None:
                rel_dist = None
            else:
                rel_dist = abs(float(prop_dist) - float(vehicle_pos))
            if isinstance(ped_type, str) and ped_type != "":
                plot_label = ped_type
            else:
                plot_label = cls_name
            mask = (df_tc["class_name"] == cls_name)
            if isinstance(ped_type, str) and ped_type != "":
                mask &= (df_tc["pedestrian_type"] == ped_type)
            if pd.notna(r["prop_distance"]):
                mask &= (df_tc["prop_distance"] == r["prop_distance"])
            imgs_with_class = df_tc[mask]["image_name"].nunique()
            efficiency = imgs_with_class / img_total if img_total > 0 else 0.0

            row = {
                "domain": domain_name,
                "source_type": source_type,
                "test_case": tc_name,
                "test_case_norm": tc_norm,
                "class_name": plot_label,
                "pedestrian_type": ped_type,
                "mean_confidence": float(r["confidence"]),
                "vehicle_position": vehicle_pos,
                "prop_distance": prop_dist,
                "relative_distance": rel_dist,
                "lab": lab,
                "car": car,
                "condition": condition,
                "images_total": img_total,
                "images_with_class": int(imgs_with_class),
                "efficiency": float(efficiency),
            }

            # append to both caller summary_rows and domain_summary_rows
            summary_rows.append(row)
            domain_summary_rows.append(row)

    # After all test case folders for this domain are processed: write domain-level summary CSV
    if domain_summary_rows:
        df_domain = pd.DataFrame(domain_summary_rows)
        domain_summary_csv_name = f"summary_{domain_name}_per_testcase_class.csv"
        domain_summary_csv_path = out_root / domain_summary_csv_name
        df_domain.to_csv(domain_summary_csv_path, index=False)
        print(
            f"[INFO] Wrote domain-level summary CSV: {domain_summary_csv_path}")


# condition runner-- supports both dry & fog layouts(excluded Medium visibility)

def run_condition(condition_folder_name: str):
    dry_base = DATA_ROOT / "input_images" / DATASET_FOLDER
    fog_base = DATA_ROOT / "input_images" / DATASET_FOLDER / condition_folder_name

    chosen_base = None
    layout = None
    if DATASET_FOLDER and DATASET_FOLDER.strip().lower() == "measurements_dry_03_11_2025".lower() and dry_base.exists():
        chosen_base = dry_base
        layout = "dry"
    elif "rain" in condition_folder_name.lower():
        chosen_base = DATA_ROOT / "input_images" / \
            DATASET_FOLDER / condition_folder_name
        layout = "rain"
    elif (DATA_ROOT / "input_images" / DATASET_FOLDER).exists() and (DATA_ROOT / "input_images" / DATASET_FOLDER / condition_folder_name).exists():
        chosen_base = DATA_ROOT / "input_images" / \
            DATASET_FOLDER / condition_folder_name
        layout = "fog"
    else:
        alt = DATA_ROOT / "input_images" / condition_folder_name
        if alt.exists():
            chosen_base = alt
            layout = "plain_condition"
        else:
            print(
                f"[WARN] Could not find input folder for condition '{condition_folder_name}'. Tried:")
            print(f"  - Dry candidate: {dry_base}")
            print(f"  - Fog candidate: {fog_base}")
            print(f"  - Alt candidate: {alt}")
            root = DATA_ROOT / "input_images"
            if root.exists():
                print(f"[INFO] Contents of {root}:")
                for c in sorted(root.iterdir()):
                    print("   ", "DIR" if c.is_dir() else "FILE", c.name)
            else:
                print(f"[INFO] {root} does not exist.")
            return

    if layout == "dry":
        input_base = chosen_base
        thermal_in = input_base / "Thermal"
        rgb_in = input_base / "RGB"
        output_base = OUTPUT_BASE
    elif layout == "rain":
        input_base = chosen_base
        thermal_in = input_base / "Thermal"
        rgb_in = input_base / "RGB"
        output_base = OUTPUT_BASE / condition_folder_name
    elif layout == "fog":
        input_base = chosen_base
        thermal_in = input_base / "Thermal"
        rgb_in = input_base / "RGB"
        output_base = OUTPUT_BASE / condition_folder_name
    else:
        input_base = chosen_base
        thermal_in = input_base / "Thermal"
        rgb_in = input_base / "RGB"
        output_base = OUTPUT_BASE / condition_folder_name

    output_base.mkdir(parents=True, exist_ok=True)
    thermal_out = output_base / "Thermal"
    rgb_out = output_base / "RGB"
    thermal_out.mkdir(parents=True, exist_ok=True)
    rgb_out.mkdir(parents=True, exist_ok=True)
    plot_dir = output_base / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] Running condition: {condition_folder_name} (layout={layout})")
    print(f"[INFO] Thermal input: {thermal_in} (exists={thermal_in.exists()})")
    print(f"[INFO] RGB input:     {rgb_in} (exists={rgb_in.exists()})")
    print(f"[INFO] Output base:   {output_base}")

    thermal_model = YOLO(THERMAL_MODEL_PATH)
    rgb_model = YOLO(RGB_MODEL_PATH)
    summary_rows = []

    if thermal_in.exists():
        process_domain(domain_name="Thermal", in_root=thermal_in,
                       out_root=thermal_out, model=thermal_model, summary_rows=summary_rows)
    else:
        print(f"[WARN] input folder missing: {thermal_in}")
    if rgb_in.exists():
        process_domain(domain_name="RGB", in_root=rgb_in,
                       out_root=rgb_out, model=rgb_model, summary_rows=summary_rows)
    else:
        print(f"[WARN] input folder missing: {rgb_in}")

    if not summary_rows:
        print(
            f"[WARN] No detections for condition {condition_folder_name} (no summary rows).")
        return

    # combined summary for the condition- all domains together
    sum_df = pd.DataFrame(summary_rows)
    summary_csv = output_base / "summary_per_testcase_class.csv"
    sum_df.to_csv(summary_csv, index=False)
    print(f"[INFO] Wrote summary CSV: {summary_csv}")

    # plots for the condition, uses combined summary rows
    plot_distance_vs_conf_per_class(
        sum_df, condition_name=condition_folder_name, plot_dir=plot_dir)
    plot_efficiency_per_class(
        sum_df, condition_name=condition_folder_name, plot_dir=plot_dir)
    print(f"[INFO] Completed condition: {condition_folder_name}")


# Main

def main():
    input_root = DATA_ROOT / "input_images"
    if not input_root.exists():
        print(f"[ERROR] input_images root not found: {input_root}")
        sys.exit(1)

    dry_base = DATA_ROOT / "input_images" / DATASET_FOLDER
    DATASET_NAME = DATASET_FOLDER.strip().lower()

    dataset_name = DATASET_FOLDER.strip().lower()

    if dry_base.exists() and "dry" in dataset_name:
        # Dry dataset runs once
        run_condition(DATASET_FOLDER)
        return

    # Rain and Fog run via CONDITIONS
    for cond in CONDITIONS:
        run_condition(cond)


if __name__ == "__main__":
    main()
