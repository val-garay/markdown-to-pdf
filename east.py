# pinn_2d_ksoil.py  (4-soil support: CP, GR, SD, DC, with aspect)
# Train from Excel (point), predict on rasters (maps)
# 2-D Richards-style residual with SOIL-SPECIFIC Ksatx, Ksatz per soil.
# LULC uses ONLY FVC (Fractional Vegetation Cover) and ISA (Impervious Surface Area).
# Prediction REQUIRES a soil thickness map (soil/soil_thickness.tif).
# Soil type supports 4 classes: CP, GR, SD, DC (one-hot); physics & model blend parameters per-pixel.
# Slope for TRAINING is constant per file and taken from the filename, e.g. siteA_CP_15.xlsx → slope=15°.
# TRAINING RAINFALL: expects 'rain_mm_10min' (mm/10min). Backward compatible with 'rain_mm_h' (auto-converted /6).
# PREDICTION RAIN RASTERS: expected in mm/h (NO conversion).
# NEW: Aspect (azimuth, degrees) is included as an 18th input feature (column index 17).
#      It can come from Excel row 'aspect_deg'/'aspect', or from Aspect.tif, or default 0°.

import os, warnings, re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -------------------- Optional rasterio --------------------
try:
    import rasterio
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False
    print("Warning: rasterio not available. GeoTIFF I/O will be skipped.")

THICKNESS_TRAIN_NORM = 3.0  # meters (must match soil_thickness_m_train in traine

def safe_mse(x: torch.Tensor) -> torch.Tensor:
    """MSE that ignores non-finite values (NaN/Inf)."""
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return torch.zeros((), device=x.device)
    return torch.mean(x ** 2)

def aggregate_rain_to_moisture_time(
    t_moist_sec: np.ndarray,
    t_rain_sec: np.ndarray,
    rain_values: np.ndarray,
    rain_unit: str = "mm_10min",   # "mm_10min" or "mm_h"
):
    """
    Aggregate rainfall to match moisture observation intervals.

    Parameters
    ----------
    t_moist_sec : (T,) array
        Moisture timestamps [seconds]
    t_rain_sec : (R,) array
        Rainfall timestamps [seconds]
    rain_values : (R,) array
        Rainfall values
    rain_unit : str
        "mm_10min" → rain_values are depths per rain interval
        "mm_h"     → rain_values are intensities (mm/h)

    Returns
    -------
    rain_mm_step : (T,) array
        Rainfall depth per moisture interval [mm]
    dt_sec : (T,) array
        Moisture interval duration [s]
    rain_mm10_equiv : (T,) array
        Equivalent mm/10min (for NN input feature)
    """
    t_moist_sec = np.asarray(t_moist_sec, dtype=float)
    t_rain_sec = np.asarray(t_rain_sec, dtype=float)
    rain_values = np.asarray(rain_values, dtype=float)

    if t_moist_sec.ndim != 1 or t_rain_sec.ndim != 1:
        raise ValueError("t_moist_sec and t_rain_sec must be 1D arrays.")

    if len(t_moist_sec) < 2:
        raise ValueError("Need at least 2 moisture timestamps to compute dt.")

    # Moisture Δt (length T)
    dt_sec = np.diff(t_moist_sec)
    dt_sec = np.concatenate([[dt_sec[0]], dt_sec])  # same length as moisture

    # Guard against non-positive dt
    dt_sec = np.where(dt_sec > 0, dt_sec, np.nanmedian(dt_sec[dt_sec > 0]))

    rain_mm_step = np.zeros_like(t_moist_sec, dtype=float)

    # Precompute rain interval ends
    if len(t_rain_sec) == 1:
        # Single rain sample: assume it applies with same dt as moisture median
        r_dt = float(np.nanmedian(dt_sec))
        r_starts = t_rain_sec.copy()
        r_ends = t_rain_sec + r_dt
    else:
        # Interval end for j is next timestamp; last uses last dt
        r_dt_last = t_rain_sec[-1] - t_rain_sec[-2]
        r_dt_last = r_dt_last if r_dt_last > 0 else float(np.nanmedian(dt_sec))
        r_starts = t_rain_sec
        r_ends = np.concatenate([t_rain_sec[1:], [t_rain_sec[-1] + r_dt_last]])

    # Loop moisture intervals and accumulate overlap contribution
    for i in range(len(t_moist_sec)):
        t0 = t_moist_sec[i]
        t1 = t0 + dt_sec[i]

        for j in range(len(r_starts)):
            r0 = r_starts[j]
            r1 = r_ends[j]

            overlap = max(0.0, min(t1, r1) - max(t0, r0))
            if overlap <= 0:
                continue

            if rain_unit == "mm_10min":
                # rain_values[j] is depth over interval (r1-r0)
                denom = max(r1 - r0, 1e-9)
                frac = overlap / denom
                rain_mm_step[i] += rain_values[j] * frac

            elif rain_unit == "mm_h":
                # rain_values[j] is intensity
                rain_mm_step[i] += rain_values[j] * overlap / 3600.0

            else:
                raise ValueError("rain_unit must be 'mm_10min' or 'mm_h'")

    # Convert mm per step to equivalent mm/10min feature
    rain_mmh = rain_mm_step * 3600.0 / np.maximum(dt_sec, 1e-6)
    rain_mm10_equiv = rain_mmh / 6.0

    return rain_mm_step, dt_sec, rain_mm10_equiv

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

INPUT_FEATURES = [
    "t",             # 0
    "x",             # 1
    "z",             # 2
    "slope_deg",     # 3
    "rain_mm_10min", # 4
    "runoff",        # 5
    "FVC",           # 6
    "ISA",           # 7
    "soil_CP",       # 8
    "soil_GR",       # 9
    "soil_SD",       # 10
    "soil_DC",       # 11
    "soil_SC",       # 12
    "temp_C",        # 13
    "RH_pct",        # 14
    "wind_ms",       # 15
    "wind_dir_deg",  # 16
    "soil_thickness",# 17
    "aspect_deg",    # 18
    "theta0",        # 19
]
# len(INPUT_FEATURES) must be 20

# -------------------- AHP weights (UPDATED) --------------------
# Put your actual AHP weights here
AHP_WEIGHTS = {
    "rain_mm_10min": 0.438,
    "slope_deg":     0.296,
    "aspect_deg":    0.102,
    "FVC":           0.082,
    "ISA":           0.082,
}

def build_penalty_factors(
    feature_names,
    ahp_weights,
    lambda_base: float = 1e-3,
    eps: float = 1e-3,
) -> torch.Tensor:
    lambdas = []
    for name in feature_names:
        w = ahp_weights.get(name, None)

        if w is None:
            # no AHP weight -> neutral penalty
            lam_j = lambda_base
        else:
            phi_j = 1.0 / (w + eps)
            lam_j = lambda_base * phi_j
        lambdas.append(lam_j)

    return torch.tensor(lambdas, dtype=torch.float32)

# ==========================================================
# Soil parameters WITH soil-specific directional Ksat (m/s)
# Now supports 5 classes: CP, GR, SD, DC, SC (Lab scale)
# ==========================================================
class SoilParameters:
    # Carboniferous Permian (CP)
    CP = {'theta_s':0.56, 'theta_r':0.016, 'alpha':0.018, 'n':1.17, 'm':0.145,  'Ksatx':4.94e-8, 'Ksatz':4.94e-8}
    # SC (Lab scale)
    SC = {'theta_s':0.37, 'theta_r':0.01, 'alpha':0.0290, 'n':1.492, 'm':0.329,  'Ksatx':0.000142, 'Ksatz':0.000142}
    # Silurian Devonian (SD)
    SD = {'theta_s':0.43, 'theta_r':0.013, 'alpha':0.811, 'n':1.48, 'm':0.324,'Ksatx':1.73e-5,'Ksatz':1.73e-5}
    # Devonian Carboniferous (DC)
    DC = {'theta_s':0.42, 'theta_r':0.016, 'alpha':0.787, 'n':1.55, 'm':0.355,'Ksatx':2.05e-5,'Ksatz':2.05e-5}
    # Granite (GR)
    GR = {'theta_s':0.48, 'theta_r':0.024, 'alpha':0.0035, 'n':1.36, 'm':0.265,'Ksatx':9.78e-7,'Ksatz':9.78e-7}

    TABLE: Dict[str, Dict[str, float]] = {
        'CP': CP, 'GR': GR, 'SD': SD, 'DC': DC, 'SC': SC,
    }
    ORDER = ['CP','GR','SD','DC', 'SC']  # one-hot channel order

    # enforce m = 1 - 1/n if not provided
    for k in TABLE:
        if ('m' not in TABLE[k]) or (TABLE[k]['m'] is None):
            TABLE[k]['m'] = 1.0 - 1.0 / TABLE[k]['n']

    @staticmethod
    def stack_param(name: str, device_:torch.device=None) -> torch.Tensor:
        """Return a (5,) tensor of the parameter in ORDER = [CP,GR,SD,DC,SC]."""
        arr = [SoilParameters.TABLE[tag][name] for tag in SoilParameters.ORDER]
        t = torch.tensor(arr, dtype=torch.float32)
        return t.to(device_) if device_ is not None else t

# ==========================================================
# LULC: FVC and ISA (fallback generator if rasters missing)
# ==========================================================
class LandUseGenerator:
    @staticmethod
    def generate_landuse_from_slope(slope:np.ndarray)->Dict[str,np.ndarray]:
        fvc = np.zeros_like(slope, dtype=np.float32)
        isa = np.zeros_like(slope, dtype=np.float32)
        flat   = slope < 5
        gentle = (slope>=5)&(slope<15)
        steep  = slope>=15
        fvc[flat]   = 0.7; isa[flat]   = 0.2
        fvc[gentle] = 0.8; isa[gentle] = 0.1
        fvc[steep]  = 0.4; isa[steep]  = 0.05
        return {'FVC': fvc, 'ISA': isa}

# ==========================================================
# Data loader (Excel for training; rasters for prediction)
# ==========================================================
class DataLoader:
    def __init__(self, project_dir:str):
        self.project_dir = project_dir
        # Grids
        self.dem_data = None
        self.slope_data = None
        self.slope_meta = None
        self.slope_transform = None
        self.pixel_size_x = 1.0

        # Aspect grid (deg, azimuth)
        self.aspect_data = None

        # Time stacks
        self.rainfall_maps = {}          # maps are in mm/h for prediction
        self.met_maps = {'temp_C':{}, 'RH_pct':{}, 'wind_ms':{}, 'wind_dir_deg':{}}

        # Static LULC arrays (maps)
        self.landuse_arrays = None

        # Soil maps
        self.soil_thickness_arr = None       # 2D (m) REQUIRED for prediction
        self.soil_type_oh = None             # (H,W,5) one-hot [CP,GR,SD,DC,SC]

        # Excel (for training only)
        self.excel_data = {}

    def _load_time_stack(self, folder:str, prefix:str)->dict:
        d = {}
        if not HAS_RASTERIO:
            return d
        base = os.path.join(self.project_dir, folder)
        if not os.path.isdir(base):
            return d
        files = [f for f in os.listdir(base) if f.startswith(prefix+"_t") and f.endswith(".tif")]
        files.sort()
        for fn in files:
            try:
                t = int(fn.split("_t")[-1].replace(".tif",""))
                with rasterio.open(os.path.join(base, fn)) as src:
                    arr = src.read(1).astype(np.float32)
                d[t] = arr
            except Exception as e:
                print(f"Warning: cannot load {folder}/{fn}: {e}")
        return d

    # -------------------- rasters --------------------
    def load_dem(self) -> bool:
        """Load DEM.tif (meters). Optional but recommended for routing/runon."""
        if not HAS_RASTERIO:
            return False
        p = os.path.join(self.project_dir, "DEM.tif")
        if not os.path.exists(p):
            print("DEM.tif not found; flow routing (runon) will be skipped.")
            return False
        try:
            with rasterio.open(p) as src:
                self.dem_data = src.read(1).astype(np.float32)
            print("Loaded DEM:", self.dem_data.shape)
            return True
        except Exception as e:
            print("Error loading DEM:", e)
            return False

    def load_slope(self)->bool:
        if not HAS_RASTERIO:
            print("rasterio missing; cannot load Slope.tif")
            return False
        p = os.path.join(self.project_dir, "Slope.tif")
        if not os.path.exists(p):
            print("Slope.tif not found at", p)
            return False
        try:
            with rasterio.open(p) as src:
                self.slope_data = src.read(1).astype(np.float32)
                self.slope_meta = src.meta.copy()
                self.slope_transform = src.transform
                try:
                    self.pixel_size_x = float(abs(src.transform.a))
                except Exception:
                    self.pixel_size_x = 1.0
            print("Loaded slope:", self.slope_data.shape)
            return True
        except Exception as e:
            print("Error loading slope:", e)
            return False

    def load_aspect(self)->bool:
        """Load Aspect.tif (azimuth, degrees). Optional for prediction."""
        if not HAS_RASTERIO:
            return False
        p = os.path.join(self.project_dir, "Aspect.tif")
        if not os.path.exists(p):
            print("Aspect.tif not found; aspect will be taken from Excel or default 0°.")
            return False
        try:
            with rasterio.open(p) as src:
                self.aspect_data = src.read(1).astype(np.float32)
            print("Loaded aspect:", self.aspect_data.shape)
            return True
        except Exception as e:
            print("Error loading aspect:", e)
            return False

    def load_rain_maps(self)->bool:
        """
        Rain rasters are assumed in mm/h for prediction (NO conversion to mm/10min here).
        """
        self.rainfall_maps = {}
        if not HAS_RASTERIO:
            return False
        base = os.path.join(self.project_dir, "rain")
        if not os.path.isdir(base):
            print("No 'rain' folder; map prediction will fallback to zeros.")
            return False
        files = [f for f in os.listdir(base) if f.startswith("rain_t") and f.endswith(".tif")]
        files.sort()
        for fn in files:
            try:
                t = int(fn.replace("rain_t","").replace(".tif",""))
                with rasterio.open(os.path.join(base, fn)) as src:
                    # keep mm/h as-is
                    self.rainfall_maps[t] = src.read(1).astype(np.float32)
            except Exception as e:
                print("Warning rain load:", e)
        print(f"Loaded rain maps: {len(self.rainfall_maps)} (unit = mm/h)")
        return len(self.rainfall_maps) > 0

    def load_met_maps(self)->bool:
        self.met_maps['temp_C']       = self._load_time_stack("met","temp")
        self.met_maps['RH_pct']       = self._load_time_stack("met","rh")
        self.met_maps['wind_ms']      = self._load_time_stack("met","wind_ms")
        self.met_maps['wind_dir_deg'] = self._load_time_stack("met","wind_dir")
        ntemp = len(self.met_maps['temp_C'])
        nrh   = len(self.met_maps['RH_pct'])
        nms   = len(self.met_maps['wind_ms'])
        nd    = len(self.met_maps['wind_dir_deg'])
        print(f"Loaded met stacks | temp:{ntemp} rh:{nrh} wind_ms:{nms} wind_dir:{nd}")
        return any([ntemp, nrh, nms, nd])

    def load_lulc_rasters(self)->bool:
        if not HAS_RASTERIO:
            return False
        base = os.path.join(self.project_dir,"lulc")
        paths = {
            'FVC': os.path.join(base,"fvc.tif"),
            'ISA': os.path.join(base,"isa.tif")
        }
        if not all(os.path.exists(p) for p in paths.values()):
            print("LULC rasters (FVC/ISA) missing; will fallback to slope-based synthetic.")
            return False
        try:
            arrs = {}
            for k,p in paths.items():
                with rasterio.open(p) as src:
                    a = src.read(1).astype(np.float32)
                    arrs[k] = np.clip(a, 0.0, 1.0)
            self.landuse_arrays = arrs
            print("Loaded LULC rasters: FVC/ISA.")
            return True
        except Exception as e:
            print("LULC raster load error:", e)
            return False

    def load_soil_thickness(self)->bool:
        if not HAS_RASTERIO:
            return False
        p = os.path.join(self.project_dir, "soil","soil_thickness.tif")
        if not os.path.exists(p):
            print("ERROR: soil_thickness.tif not found; this is REQUIRED for prediction.")
            return False
        with rasterio.open(p) as src:
            arr = src.read(1).astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, 2.0).astype(np.float32)
        self.soil_thickness_arr = arr
        print("Loaded soil thickness:", arr.shape)
        return True

    # Load soil type codes (0=CP,1=GR,2=SD,3=DC,4=SC) and convert to one-hot (H,W,5)
    def load_soil_type(self)->bool:
        if not HAS_RASTERIO:
            return False
        p = os.path.join(self.project_dir, "soil","soil_type.tif")
        if not os.path.exists(p):
            print("Info: soil_type.tif not found; will use single soil type later.")
            return False
        with rasterio.open(p) as src:
            codes = src.read(1).astype(np.int32)
        H, W = codes.shape
        oh = np.zeros((H, W, 5), dtype=np.float32)
        for code, ch in [(0,0),(1,1),(2,2),(3,3),(4,4)]:
            oh[:, :, ch] = (codes == code).astype(np.float32)
        s = np.clip(oh.sum(axis=2, keepdims=True), 1e-6, None)
        self.soil_type_oh = oh / s
        print("Loaded soil type map (one-hot):", self.soil_type_oh.shape)
        return True

    # -------------------- Excel (training only) --------------------
    def load_excel(self) -> bool:
        """
        Recursively load training *.xlsx from project_dir and project_dir/train/.
        Each file name defines soil type and constant slope, e.g.:
            GR_s30_90mmh.xlsx → soil=GR, slope ≈ 30°

        Sheet rows (labels col A, values across time in cols B..):
          REQUIRED:
            - time row: 'time_sec', 'time', or anything containing 'time'
            - rainfall row: 'rain_mm_10min' OR 'rain_mm_h'
            - depth rows: either 'depth_cm=<number>' OR numeric labels like 5, 15, 30 (cm)
          OPTIONAL:
            - temp_C, RH_pct, wind_ms, wind_dir_deg, fvc, isa
            - aspect_deg or aspect (azimuth in degrees)
        """
        def collect_xlsx(root):
            paths = []
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith(".xlsx"):
                        paths.append(os.path.join(dirpath, fn))
            return paths

        # collect from project_dir and project_dir/train
        xl_files = set(collect_xlsx(self.project_dir))
        train_dir = os.path.join(self.project_dir, "train")
        if os.path.isdir(train_dir):
            xl_files.update(collect_xlsx(train_dir))

        xl_files = sorted(list(xl_files))
        if not xl_files:
            print("No Excel file found for training.")
            return False

        def row_vals(row):
            vals = pd.to_numeric(row[1:], errors='coerce').to_numpy(dtype='float64')
            return None if np.isnan(vals).all() else vals

        used = 0
        self.excel_data = {}

        for path in xl_files:
            excel_file = os.path.basename(path)
            name_upper = excel_file.upper().replace(".XLSX", "")

            # --- extract soil tag from filename: GR_s30_10mmh.xlsx → GR ---
            soil_tag = None
            tokens = re.split(r'[^A-Z0-9]+', name_upper)
            tokens = [t for t in tokens if t]

            for tag in SoilParameters.ORDER:  # ['CP','GR','SD','DC','SC']
                if tag in tokens:
                    soil_tag = tag
                    break
            if soil_tag is None:
                print(f"⚠️ Cannot detect soil type in filename: {excel_file}; default=CP")
                soil_tag = "CP"

            # slope from 's0', 's30', etc.
            slope_deg = parse_slope_from_filename(excel_file, default=15.0)

            one_hot = np.zeros((5,), dtype=np.float32)
            one_hot[SoilParameters.ORDER.index(soil_tag)] = 1.0

            # --- read Excel content ---
            try:
                df = pd.read_excel(path, header=None)
            except Exception as e:
                print(f"Skip {excel_file}: cannot open ({e}).")
                continue

            time_vals = None
            rain_10min = None        # preferred (mm/10min)
            rain_h_intensity = None  # legacy (mm/h)

            met  = {'temp_C': None, 'RH_pct': None, 'wind_ms': None, 'wind_dir_deg': None}
            lulc = {'fvc': None, 'isa': None}
            aspect = None
            depths_m, moist_rows = [], []

            # keys that are NOT depth rows
            non_depth_keys = {
                'time', 'time_sec',
                'rain_mm_10min', 'rain_mm_h',
                'temp_c', 'rh_pct', 'wind_ms', 'wind_dir_deg',
                'fvc', 'isa',
                'aspect_deg', 'aspect'
            }

            for r in range(len(df)):
                label = df.iloc[r, 0]
                row   = df.iloc[r, :]

                # --- string label ---
                if isinstance(label, str):
                    key = label.strip().lower()

                    # time row: accept 'time', 'time_sec', 'time (s)', etc.
                    if key in ('time', 'time_sec') or 'time' in key:
                        time_vals = row_vals(row)

                    elif key == 'rain_mm_10min':
                        rain_10min = row_vals(row)

                    elif key == 'rain_mm_h':
                        rain_h_intensity = row_vals(row)

                    elif key in ('temp_c', 'rh_pct', 'wind_ms', 'wind_dir_deg'):
                        v = row_vals(row)
                        if key == 'temp_c':
                            norm = 'temp_C'
                        elif key == 'rh_pct':
                            norm = 'RH_pct'
                        else:
                            norm = key
                        met[norm] = v

                    elif key in ('fvc', 'isa'):
                        lulc[key] = row_vals(row)

                    elif key in ('aspect_deg', 'aspect'):
                        aspect = row_vals(row)

                    elif key.startswith('depth_cm='):
                        # e.g. "depth_cm=5"
                        try:
                            dcm = float(key.split('=')[1])
                            depths_m.append(dcm / 100.0)
                            moist_rows.append(row_vals(row))
                        except Exception as e:
                            print(f"Warn {excel_file}: bad depth label '{key}' ({e}).")

                    # we still allow numeric depth parsing below, so don't 'continue' here

                # --- numeric depth labels, e.g. 5, 15, 30 in col A ---
                key = str(label).strip().lower() if isinstance(label, str) else None
                if key not in non_depth_keys:
                    try:
                        dcm = float(label)   # numeric or numeric string
                        depths_m.append(dcm / 100.0)
                        moist_rows.append(row_vals(row))
                    except Exception:
                        # not a numeric depth, ignore
                        pass

            # required: time, rainfall, at least 1 depth row
            if time_vals is None or (rain_10min is None and rain_h_intensity is None) or len(depths_m) == 0:
                print(f"Skip {excel_file}: missing time/rain/depth.")
                continue

            # -----------------------------
            # Moisture time grid
            # -----------------------------
            t_moist = np.asarray(time_vals, dtype=np.float64)
            T = len(t_moist)

            def ensure_T(v, name):
                if v is None:
                    return None
                v = np.asarray(v, dtype='float64')
                if len(v) != T:
                    raise ValueError(f"{name} length {len(v)} != T={T}")
                return v

            # -----------------------------
            # Build a rain time grid (can be different length from moisture)
            # -----------------------------
            if rain_h_intensity is not None:
                rain_vals = np.asarray(rain_h_intensity, dtype=np.float64)
                rain_unit = "mm_h"
            else:
                rain_vals = np.asarray(rain_10min, dtype=np.float64)
                rain_unit = "mm_10min"

            if len(rain_vals) == T:
                t_rain = t_moist.copy()
            else:
                dt_r = 600.0
                t_rain = t_moist[0] + np.arange(len(rain_vals), dtype=np.float64) * dt_r

            # -----------------------------
            # Aggregate rainfall onto moisture intervals
            # -----------------------------
            rain_mm_step, dt_sec, rain_mm10_equiv = aggregate_rain_to_moisture_time(
                t_moist_sec=t_moist,
                t_rain_sec=t_rain,
                rain_values=rain_vals,
                rain_unit=rain_unit,
            )

            rain_10min_feature = rain_mm10_equiv  # NN feature (mm/10min-equivalent)

            # -----------------------------
            # Optional series (force length T)
            # -----------------------------
            def prep(v, default):
                if v is None:
                    return np.full(T, default, dtype='float64')
                v = np.asarray(v, dtype='float64')
                if len(v) != T:
                    raise ValueError("Optional series length mismatch.")
                return v

            met_series = {
                'temp_C':       prep(met['temp_C'], 25.0),
                'RH_pct':       prep(met['RH_pct'], 70.0),
                'wind_ms':      prep(met['wind_ms'], 1.0),
                'wind_dir_deg': prep(met['wind_dir_deg'], 0.0),
            }

            def clip01(a):
                return np.clip(np.asarray(a, dtype='float64'), 0.0, 1.0)

            FVC = clip01(prep(lulc['fvc'], 0.5))
            ISA = clip01(prep(lulc['isa'], 0.2))
            lulc_series = {'FVC': FVC, 'ISA': ISA}

            aspect_series = prep(aspect, 0.0)

            # moisture matrix must match T
            moist_matrix = []
            for i, vals in enumerate(moist_rows):
                vals = ensure_T(vals, f"moist_row[{i}]")
                moist_matrix.append(vals)
            moist_matrix = np.vstack(moist_matrix)

            # sort depths
            depths_m_arr = np.asarray(depths_m, dtype='float64')
            order = np.argsort(depths_m_arr)
            depths_m_arr = depths_m_arr[order]
            moist_matrix = moist_matrix[order, :]

            key_rel = os.path.relpath(path, self.project_dir)
            self.excel_data[key_rel] = {
                'time':         np.asarray(t_moist, dtype='float64'),
                'dt_sec':       np.asarray(dt_sec, dtype='float64'),            # ✅ NEW
                'rain_mm_step': np.asarray(rain_mm_step, dtype='float64'),      # ✅ NEW
                'rainfall':     np.asarray(rain_10min_feature, dtype='float64'),# ✅ NN input
                'depths':       depths_m_arr,
                'moisture':     moist_matrix,
                'met_series':   met_series,
                'lulc_series':  lulc_series,
                'aspect_deg':   np.asarray(aspect_series, dtype='float64'),
                'slope_deg':    np.full(T, float(slope_deg), dtype='float64'),
                'soil_one_hot': one_hot,
                'file_name':    key_rel,
            }

            used += 1
            print(
                f"Loaded Excel: {key_rel} | soil={soil_tag}, "
                f"slope={slope_deg:.1f}°, T={T}, depths={depths_m_arr.tolist()}"
            )

        print(f"Excel training files loaded: {used}/{len(xl_files)}")
        return used > 0

    # -------------------- LULC arrays (prediction) --------------------
    def build_lulc_arrays(self)->bool:
        if self.load_lulc_rasters():
            return True
        if self.slope_data is None:
            print("Need slope to build synthetic LULC.")
            return False
        self.landuse_arrays = LandUseGenerator.generate_landuse_from_slope(self.slope_data)
        print("Built synthetic FVC/ISA from slope (fallback).")
        return True

    # -------------------- master load --------------------
    def load_all(self)->bool:
        ok = True
        ok &= self.load_slope()
        self.load_dem()
        self.load_aspect()               # optional aspect grid
        ok &= self.load_rain_maps()      # mm/h for prediction rasters
        self.load_met_maps()             # optional but recommended
        ok &= self.build_lulc_arrays()
        ok &= self.load_soil_thickness() # REQUIRED for prediction
        self.load_soil_type()            # optional soil-type map
        ok &= self.load_excel()          # for training
        try:
            if self.rainfall_maps:
                ks = sorted(self.rainfall_maps.keys())
                print("Rain time steps (sample):", ks[:5], "… total", len(ks))
            for k,v in self.met_maps.items():
                if v:
                    kv = sorted(v.keys())
                    print(f"Met {k} steps (sample):", kv[:5], "… total", len(kv))
        except Exception:
            pass
        return ok


# ==========================================================
# Simple D8-like runoff (placeholder)
# ==========================================================
class RunoffCalculator:
    """
    Defines runoff properly as LOCAL rainfall-excess runoff (generation),
    not accumulated inflow.

    - runoff_gen = max(rain - infil_capacity, 0)
    - Optional: route runoff_gen downslope using DEM to compute runon/accumulated flow.
    """
    def __init__(self, dem: Optional[np.ndarray] = None, nodata: float = np.nan):
        self.dem = dem
        self.nodata = nodata
        self._receiver = None  # receiver index per cell (flattened), -1 if pit/edge
        self._topo = None      # topological order for accumulation (flattened indices)
        if dem is not None:
            self._build_d8_from_dem(dem)

    @staticmethod
    def _infil_capacity_mm10(
        soil_w: np.ndarray,         # (...,5) one-hot/weights
        FVC: np.ndarray,            # (...) 0..1
        ISA: np.ndarray,            # (...) 0..1
        slope_deg: np.ndarray,      # (...) degrees
        cap_scale: float = 1.0
    ) -> np.ndarray:
        """
        Simple infiltration-capacity proxy in mm/10min.
        Uses soil Ksatz (saturated vertical conductivity) and scales by land cover + slope.

        cap ≈ Ksatz * 600s * 1000 (mm/10min) * (1-ISA) * (0.3 + 0.7*FVC) * f(slope)
        """
        # soil Ksatz per class
               # soil Ksatz per class
        Ksatz_vec = np.array([
            SoilParameters.CP['Ksatz'],
            SoilParameters.GR['Ksatz'],
            SoilParameters.SD['Ksatz'],
            SoilParameters.DC['Ksatz'],
            SoilParameters.SC['Ksatz'],
        ], dtype=np.float32)  # (5,)

        # blend Ksatz
        Ksatz = (soil_w * Ksatz_vec[None, :]).sum(axis=-1).astype(np.float32)  # (...)

        # base capacity in mm/10min
        cap0 = Ksatz * 600.0 * 1000.0  # m/s * s * mm/m = mm/10min

        # land cover scalings
        f_isa = np.clip(1.0 - ISA, 0.0, 1.0)
        f_fvc = 0.3 + 0.7 * np.clip(FVC, 0.0, 1.0)

        # slope scaling (steeper -> more runoff, less infil) : 1 / (1 + tan(slope))
        srad = np.deg2rad(np.clip(slope_deg, 0.0, 89.0))
        f_slope = 1.0 / (1.0 + np.tan(srad))

        cap = cap_scale * cap0 * f_isa * f_fvc * f_slope
        return np.clip(cap, 0.0, None).astype(np.float32)

    @classmethod
    def runoff_generation_mm10(
        cls,
        rain_mm10: np.ndarray,
        soil_w: np.ndarray,
        FVC: np.ndarray,
        ISA: np.ndarray,
        slope_deg: np.ndarray,
        cap_scale: float = 1.0
    ) -> np.ndarray:
        cap = cls._infil_capacity_mm10(soil_w, FVC, ISA, slope_deg, cap_scale=cap_scale)
        return np.maximum(rain_mm10 - cap, 0.0).astype(np.float32)

    # ------------------ OPTIONAL: DEM-based D8 routing (for runon) ------------------
    def _build_d8_from_dem(self, dem: np.ndarray):
        dem = dem.astype(np.float32)
        H, W = dem.shape
        idx = np.arange(H * W, dtype=np.int32).reshape(H, W)
        receiver = np.full((H, W), -1, dtype=np.int32)

        # neighbor offsets + distances
        nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        dist = np.array([np.sqrt(2),1,np.sqrt(2),1,1,np.sqrt(2),1,np.sqrt(2)], dtype=np.float32)

        # compute receiver by steepest descent (max drop / distance)
        for i in range(1, H-1):
            for j in range(1, W-1):
                z = dem[i, j]
                if not np.isfinite(z):
                    continue
                best = -1
                best_s = 0.0
                for k,(di,dj) in enumerate(nbrs):
                    ni, nj = i+di, j+dj
                    zn = dem[ni, nj]
                    if not np.isfinite(zn):
                        continue
                    drop = z - zn
                    if drop <= 0:
                        continue
                    s = drop / dist[k]
                    if s > best_s:
                        best_s = s
                        best = idx[ni, nj]
                receiver[i, j] = best

        rec_flat = receiver.ravel()
        self._receiver = rec_flat

        # build topo order for accumulation
        N = H*W
        indeg = np.zeros(N, dtype=np.int32)
        for u in range(N):
            v = rec_flat[u]
            if v >= 0:
                indeg[v] += 1

        # Kahn topo
        q = np.where(indeg == 0)[0].tolist()
        topo = []
        while q:
            u = q.pop()
            topo.append(u)
            v = rec_flat[u]
            if v >= 0:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        self._topo = np.array(topo, dtype=np.int32)

    def route_runoff_mm10(self, runoff_gen_mm10: np.ndarray) -> np.ndarray:
        """
        Returns accumulated runon/flow (mm/10min) at each cell by routing runoff_gen downslope.
        (Not used by BC; useful for mapping).
        """
        if self.dem is None or self._receiver is None or self._topo is None:
            raise RuntimeError("DEM routing not initialized (need DEM.tif).")

        H, W = runoff_gen_mm10.shape
        acc = runoff_gen_mm10.astype(np.float32).ravel().copy()

        for u in self._topo:
            v = self._receiver[u]
            if v >= 0:
                acc[v] += acc[u]

        return acc.reshape(H, W).astype(np.float32)

# ==========================================================
# PINN model (inputs=19). Soil is 5 one-hot channels [CP,GR,SD,DC,SC].
# TUNED: Added depth embedding & skip connections for better z-learning
# ==========================================================
class EnhancedPINN(nn.Module):
    """
    PINN architecture with depth-aware features:
      - Fourier depth embedding to help learn depth-dependent patterns
      - Skip connections for depth features
      - Smooth activations (tanh or SiLU)
      - Soft bounded output using sigmoid temperature
    """
    def __init__(
        self,
        hidden: List[int] = [128, 256, 256, 128],
        activation: str = "silu",          # CHANGED: silu often works better
        out_temp: float = 8.0,             # INCREASED: less saturation
        lambda_base: float = 1e-3,
        lambda_rest: float = 1e-5,
        use_weight_norm: bool = False,
        n_depth_freqs: int = 6,            # NEW: Fourier embedding frequencies
    ):
        super().__init__()

        self.input_dim = 20
        self.output_dim = 1
        self.out_temp = float(out_temp)
        self.n_depth_freqs = n_depth_freqs

        # Depth embedding dimension: z + sin/cos for each frequency
        self.depth_embed_dim = 1 + 2 * n_depth_freqs  # 1 + 12 = 13

        # Interaction features: t*z, slope*z, rain*z, cos(slope)*z, sin(slope)*t*z, early_rain*z, rain_time_slope*z
        self.interaction_dim = 7

        # Total input dim after depth embedding and interaction features
        # Original: 19 features, but z (index 2) gets expanded + interaction terms
        self.expanded_input_dim = (self.input_dim - 1) + self.depth_embed_dim + self.interaction_dim

        # optional scaling parameters
        self.log_Ksatx_scale = nn.Parameter(torch.tensor(0.0))
        self.log_Ksatz_scale = nn.Parameter(torch.tensor(0.0))

        self.lambda_base = lambda_base
        self.lambda_rest = lambda_rest

        # Build λ_j for feature penalties (original 18 features)
        lambda_input = build_penalty_factors(
            INPUT_FEATURES,
            AHP_WEIGHTS,
            lambda_base=lambda_base,
        )
        self.register_buffer("lambda_input", lambda_input.view(-1, 1))  # [20,1]

        # choose activation
        if activation.lower() == "tanh":
            Act = nn.Tanh
        elif activation.lower() in ["silu", "swish"]:
            Act = nn.SiLU
        else:
            raise ValueError("activation must be 'tanh' or 'silu'")

        # Main network with skip connections
        self.fc1 = nn.Linear(self.expanded_input_dim, hidden[0])
        self.act1 = Act()

        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.act2 = Act()

        # Skip connection: inject depth embedding AND interaction features at middle layer
        self.fc3 = nn.Linear(hidden[1] + self.depth_embed_dim + self.interaction_dim, hidden[2])
        self.act3 = Act()

        self.fc4 = nn.Linear(hidden[2], hidden[3])
        self.act4 = Act()

        self.fc_out = nn.Linear(hidden[3], self.output_dim)

        self._init()

    def _depth_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """
        Fourier positional encoding for depth to help learn depth-dependent patterns.
        Input: z (N, 1) normalized depth [0, 1]
        Output: (N, 1 + 2*n_freqs) = [z, sin(2^0*pi*z), cos(2^0*pi*z), ..., sin(2^(n-1)*pi*z), cos(2^(n-1)*pi*z)]
        """
        embeddings = [z]
        for i in range(self.n_depth_freqs):
            freq = (2.0 ** i) * torch.pi
            embeddings.append(torch.sin(freq * z))
            embeddings.append(torch.cos(freq * z))
        return torch.cat(embeddings, dim=-1)

    def _init(self):
        # Xavier tends to be stable for tanh PINNs
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract key features
        t = x[:, 0:1]       # time (hours)
        z = x[:, 2:3]       # normalized depth [0, 1]
        slope = x[:, 3:4]   # slope (degrees)
        rain = x[:, 4:5]    # rainfall (mm/10min)

        # Create depth embedding
        z_embed = self._depth_embedding(z)  # (N, depth_embed_dim)

        # NEW: Create interaction features for slope-depth-time-rain relationships
        # These help the model learn that infiltration depth depends on these factors
        slope_rad = slope * torch.pi / 180.0
        cos_slope = torch.cos(slope_rad)  # cos(slope) affects vertical infiltration
        sin_slope = torch.sin(slope_rad)  # sin(slope) affects lateral drainage

        # Normalize time: assume max ~20 hours for training data
        t_norm = t / 20.0  # normalized time [0, 1]

        # Interaction terms
        t_z = t * z                     # time-depth interaction
        slope_z = slope / 90.0 * z      # slope-depth interaction (normalized)
        rain_z = rain / 10.0 * z        # rain-depth interaction (normalized)
        cos_slope_z = cos_slope * z     # cos(slope) * depth - physics-informed

        # NEW: Drainage dynamics on slopes
        # sin(slope) * t * z: faster drainage at depth on steeper slopes over time
        sin_slope_t_z = sin_slope * t_norm * z

        # (1 - t_norm) * rain * z: early-time rain infiltration effect at depth
        # High values mean "recent rain reaching depth" - helps distinguish wet vs dry
        early_rain_z = (1.0 - t_norm) * rain / 10.0 * z

        # Key discriminating feature: rain intensity vs time on slopes
        # High value = should be WET (heavy rain, short time)
        # Low value = should be DRY (light rain, long time)
        rain_time_ratio = (rain / 10.0) / (t_norm + 0.1)  # avoid division by zero
        rain_time_slope_z = rain_time_ratio * sin_slope * z

        interaction_features = torch.cat([t_z, slope_z, rain_z, cos_slope_z, sin_slope_t_z, early_rain_z, rain_time_slope_z], dim=-1)  # (N, 7)

        # Build expanded input
        x_no_z = torch.cat([x[:, :2], x[:, 3:]], dim=-1)  # (N, 18) - all except z
        x_expanded = torch.cat([x_no_z[:, :2], z_embed, interaction_features, x_no_z[:, 2:]], dim=-1)

        # Forward with skip connections
        h1 = self.act1(self.fc1(x_expanded))
        h2 = self.act2(self.fc2(h1))

        # Skip: inject depth embedding AND interaction features again
        h2_skip = torch.cat([h2, z_embed, interaction_features], dim=-1)
        h3 = self.act3(self.fc3(h2_skip))

        h4 = self.act4(self.fc4(h3))
        raw = self.fc_out(h4)  # ℝ

        # soil one-hot weights columns 8..12 of ORIGINAL input
        w = x[:, 8:13]  # (N,5)

        theta_r_vec = SoilParameters.stack_param("theta_r", x.device)  # (5,)
        theta_s_vec = SoilParameters.stack_param("theta_s", x.device)  # (5,)
        theta_r = (w * theta_r_vec).sum(dim=1, keepdim=True)
        theta_s = (w * theta_s_vec).sum(dim=1, keepdim=True)

        # ---- soft bounded mapping (better gradients than tanh+clamp) ----
        y = torch.sigmoid(raw / self.out_temp)
        theta = theta_r + (theta_s - theta_r) * y

        # tiny safety clamp
        eps = 1e-6
        theta = torch.clamp(theta, theta_r + eps, theta_s - eps)
        return theta

    def penalty_loss(self) -> torch.Tensor:
        """
        Simplified penalty for the new architecture
        """
        penalty = 0.0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "weight" in name:
                penalty = penalty + self.lambda_rest * torch.sum(p ** 2)
        return penalty

# ==========================================================
# Physics loss: 2D Richards-type with 4-soil blending
# + rainfall flux boundary condition at the surface (z≈0)
# ==========================================================
class PhysicsLoss2D:

    @staticmethod
    def _Hnorm(xreq: torch.Tensor) -> torch.Tensor:
        # feature 17 is soil_thickness in meters
        H = xreq[:, 17:18]
        return H.clamp(min=1e-3)

    @staticmethod
    def _soil_blend_params(xreq: torch.Tensor):
        w = xreq[:, 8:13]
        dev = xreq.device

        theta_r_vec  = SoilParameters.stack_param('theta_r', dev)
        theta_s_vec  = SoilParameters.stack_param('theta_s', dev)
        alpha_vg_vec = SoilParameters.stack_param('alpha',   dev)
        n_vec        = SoilParameters.stack_param('n',       dev)
        m_vec        = SoilParameters.stack_param('m',       dev)
        Ksatx_vec    = SoilParameters.stack_param('Ksatx',   dev)
        Ksatz_vec    = SoilParameters.stack_param('Ksatz',   dev)

        theta_r  = (w * theta_r_vec).sum(dim=1, keepdim=True)
        theta_s  = (w * theta_s_vec).sum(dim=1, keepdim=True)
        alpha_vg = (w * alpha_vg_vec).sum(dim=1, keepdim=True)
        n        = (w * n_vec).sum(dim=1, keepdim=True)
        m        = (w * m_vec).sum(dim=1, keepdim=True)
        Ksatx    = (w * Ksatx_vec).sum(dim=1, keepdim=True)
        Ksatz    = (w * Ksatz_vec).sum(dim=1, keepdim=True)

        alpha_vg = torch.clamp(alpha_vg, 1e-6, None)
        n        = torch.clamp(n, 1.01, None)
        m        = torch.clamp(m, 1e-3, 0.999)
        Ksatx    = torch.clamp(Ksatx, 1e-12, None)
        Ksatz    = torch.clamp(Ksatz, 1e-12, None)

        return theta_r, theta_s, alpha_vg, n, m, Ksatx, Ksatz

    @staticmethod
    def _vg_h_Kr(theta, theta_r, theta_s, alpha_vg, n, m):
        Se = (theta - theta_r) / (theta_s - theta_r + 1e-8)
        Se = torch.clamp(Se, 1e-3, 1.0 - 1e-3)

        Kr = torch.sqrt(Se) * (1.0 - (1.0 - Se.pow(1.0 / m)) ** m) ** 2
        h  = -1.0 / alpha_vg * ((Se.pow(-1.0 / m) - 1.0).pow(1.0 / n))
        return h, Kr

    @staticmethod
    def residual_pde_only(model, x):
        """
        Enforce:
          dθ/dt + dqx/dx + dqz/dz = 0
        with:
          qx = -Kx ( dh/dx - sinβ )
          qz = -Kz ( dh/dz - cosβ )
        where Kx=Ksatx*Kr, Kz=Ksatz*Kr
        """
        xreq = x.clone().requires_grad_(True)
        theta = model(xreq)

        grads_theta = torch.autograd.grad(
            theta, xreq, torch.ones_like(theta),
            create_graph=True, retain_graph=True
        )[0]

        # time derivative: feature t is HOURS -> convert to per-second
        dtheta_dt_hr = grads_theta[:, 0:1]
        dtheta_dt = dtheta_dt_hr / 3600.0

        slope_deg = xreq[:, 3:4]
        beta = slope_deg * torch.pi / 180.0
        sin_b = torch.sin(beta)
        cos_b = torch.cos(beta)

        theta_r, theta_s, alpha_vg, n, m, Ksatx, Ksatz = PhysicsLoss2D._soil_blend_params(xreq)
        h, Kr = PhysicsLoss2D._vg_h_Kr(theta, theta_r, theta_s, alpha_vg, n, m)

        grads_h = torch.autograd.grad(
            h, xreq, torch.ones_like(h),
            create_graph=True, retain_graph=True
        )[0]

        # dh/dx is direct (x is meters)
        dh_dx = grads_h[:, 1:2]

        # dh/dz: z input is z_norm -> convert
        dh_dznorm = grads_h[:, 2:3]
        H = PhysicsLoss2D._Hnorm(xreq).clamp(min=1e-6)
        dh_dz = dh_dznorm / H

        # directional conductivity
        Kx = (Ksatx * Kr).clamp(min=1e-12)
        Kz = (Ksatz * Kr).clamp(min=1e-12)

        qx = -Kx * (dh_dx - sin_b)
        qz = -Kz * (dh_dz - cos_b)

        # divergences
        dqx_dx = torch.autograd.grad(
            qx, xreq, torch.ones_like(qx),
            create_graph=True, retain_graph=True
        )[0][:, 1:2]

        dqz_dznorm = torch.autograd.grad(
            qz, xreq, torch.ones_like(qz),
            create_graph=True, retain_graph=True
        )[0][:, 2:3]
        dqz_dz = dqz_dznorm / H

        res = dtheta_dt + dqx_dx + dqz_dz
        return safe_mse(res)

    @staticmethod
    def residual_bc_surface(model, x_bc, dt_sec_bc):
        """
        Surface BC at z_norm=0:
          qz(z=0) = R_lim
        using dt_sec_bc to convert rain feature (mm/10min-equivalent) -> flux (m/s).
        """
        xreq = x_bc.clone().requires_grad_(True)
        dt = dt_sec_bc.to(xreq.device).to(xreq.dtype).clamp(min=1e-6).view(-1, 1)

        theta = model(xreq)

        slope_deg = xreq[:, 3:4]
        beta = slope_deg * torch.pi / 180.0
        sin_b = torch.sin(beta)
        cos_b = torch.cos(beta)

        theta_r, theta_s, alpha_vg, n, m, Ksatx, Ksatz = PhysicsLoss2D._soil_blend_params(xreq)
        h, Kr = PhysicsLoss2D._vg_h_Kr(theta, theta_r, theta_s, alpha_vg, n, m)

        grads_h = torch.autograd.grad(
            h, xreq, torch.ones_like(h),
            create_graph=True, retain_graph=True
        )[0]

        dh_dx = grads_h[:, 1:2]
        dh_dznorm = grads_h[:, 2:3]
        H = PhysicsLoss2D._Hnorm(xreq).clamp(min=1e-6)
        dh_dz = dh_dznorm / H

        Kx = (Ksatx * Kr).clamp(min=1e-12)
        Kz = (Ksatz * Kr).clamp(min=1e-12)

        qz = -Kz * (dh_dz - cos_b)

        # features are mm/10min-equivalent
        rain_mm10eq   = xreq[:, 4:5]
        runoff_mm10eq = xreq[:, 5:6]

        # mm over dt
        rain_mm_step   = rain_mm10eq   * (dt / 600.0)
        runoff_mm_step = runoff_mm10eq * (dt / 600.0)

        # m/s
        rain_m_s   = (rain_mm_step   * 1e-3) / dt
        runoff_m_s = (runoff_mm_step * 1e-3) / dt

        R_eff = torch.clamp(rain_m_s - runoff_m_s, min=0.0)
        R_lim = torch.minimum(R_eff, Kz)  # infiltration limited by Kz

        bc_res = (qz - R_lim) * 1e4
        finite = torch.isfinite(bc_res)
        if finite.any():
            return torch.mean(bc_res[finite] ** 2)
        return torch.zeros((), device=xreq.device)

# ==========================================================
# PINN loss block (data + physics + vertical regularization + penalty LR)
# ==========================================================
def pinn_loss_block(
    model,
    X_batch,
    Y_batch,
    epoch: int,
    dt_sec_batch=None,
    collocation_X=None,
    device=device
):
    """
    X columns (20):
      0  t_hr (hours)
      1  x_m  (meters)
      2  z_norm (–)                  z_norm = z_m / thickness_train_norm
      3  slope_deg
      4  rain_mm_10min
      5  runoff_mm_10min
      6  FVC
      7  ISA
      8..12 soil one-hot [CP,GR,SD,DC,SC]
      13 temp_C
      14 RH_pct
      15 wind_ms
      16 wind_dir_deg
      17 soil_thickness_m (meters)
      18 aspect_deg
      19 theta0
    """
    X_batch = X_batch.to(device)
    Y_batch = Y_batch.to(device)

    # need gradients wrt inputs
    X_batch.requires_grad_(True)

    # forward
    Y_pred = model(X_batch)

    # --- profile-flatness penalty per time (cheap, works) ---
    t_vals = X_batch[:, 0:1]
    # discretize time to nearest 10-min step in hours
    dt_hr = 600.0 / 3600.0
    t_bin = torch.round(t_vals / dt_hr) * dt_hr

    profile_loss = torch.tensor(0.0, device=device)
    uniq = torch.unique(t_bin)

    for tb in uniq:
        m = (t_bin == tb).squeeze(1)
        if m.sum() < 5:
            continue
        # variance across depth at this time
        var_pred = torch.var(Y_pred[m])
        var_obs  = torch.var(Y_batch[m])
        if var_obs > 1e-6:
            profile_loss = profile_loss + F.relu(var_obs - var_pred)

    profile_flatness_loss = 5.0 * (profile_loss / (len(uniq) + 1e-6))


    # ------------------------
    # DEPTH-WEIGHTED data losses (give more weight to deeper points)
    # ------------------------
    z_norm = X_batch[:, 2:3]  # normalized depth [0, 1]
    t_hr = X_batch[:, 0:1]    # time in hours
    slope_deg = X_batch[:, 3:4]  # slope in degrees

    # Weight increases with depth: w = 1 + 2*z_norm (so 30cm gets 3x weight vs surface)
    depth_weights = 1.0 + 2.0 * z_norm

    # Combine weights (just depth weighting - removed time weighting to avoid dry bias)
    combined_weights = depth_weights
    combined_weights = combined_weights / combined_weights.mean()  # normalize

    # Weighted MSE and MAE
    sq_err = (Y_pred - Y_batch) ** 2
    abs_err = torch.abs(Y_pred - Y_batch)
    mse = torch.mean(combined_weights * sq_err)
    mae = torch.mean(combined_weights * abs_err)

    # Also add per-depth-bin losses to ensure each depth level is learned
    # Bin depths: shallow (z<0.3), mid (0.3<=z<1.0), deep (z>=1.0)
    z_m = X_batch[:, 2:3] * float(THICKNESS_TRAIN_NORM)
    zmax = torch.max(z_m).clamp(min=1e-6)

    shallow_mask = z_m < 0.30
    mid_mask     = (z_m >= 0.30) & (z_m < 1.00)
    deep_mask    = z_m >= 1.00


    mse_shallow = torch.mean(sq_err[shallow_mask]) if shallow_mask.any() else torch.tensor(0.0, device=device)
    mse_mid = torch.mean(sq_err[mid_mask]) if mid_mask.any() else torch.tensor(0.0, device=device)
    mse_deep = torch.mean(sq_err[deep_mask]) if deep_mask.any() else torch.tensor(0.0, device=device)

    # NEW: Slope-aware deep layer loss - extra penalty for slope>0 at depth
    slope_deep_mask = deep_mask & (X_batch[:, 3:4] > 10.0)
    mse_slope_deep = torch.mean(sq_err[slope_deep_mask]) if slope_deep_mask.any() else torch.tensor(0.0, device=device)

    # Per-depth loss with emphasis on deep layer AND sloped deep points
    depth_bin_loss = 0.5 * mse_shallow + 1.0 * mse_mid + 2.0 * mse_deep + 3.0 * mse_slope_deep

    # ------------------------
    # Slope-specific loss: directly target slope=30° deep layer cases
    # ------------------------
    slope_deg = X_batch[:, 3:4]  # slope in degrees

    # SLOPE-SPECIFIC direct MSE loss with very high weight
    slope_deep_mask = (slope_deg > 15) & deep_mask
    if slope_deep_mask.any():
        slope_mse = torch.mean((Y_batch[slope_deep_mask] - Y_pred[slope_deep_mask]) ** 2)
    else:
        slope_mse = torch.tensor(0.0, device=device)

    wetdry_loss = 20.0 * slope_mse  # very high weight on slope cases

    # ------------------------
    # Physics loss (PDE + BC)
    # ------------------------
    # BC points at surface: z_norm=0
    Xc_bc = X_batch.detach().clone()
    Xc_bc[:, 2:3] = 0.0
    Xc_bc.requires_grad_(True)

    # Interior points: sample z_norm in (0,1), biased toward surface
    Xc_int = X_batch.detach().clone()
    u = torch.rand_like(Xc_int[:, 2:3])
    z_norm = 0.5 * u + 0.5 * (u * u)  # more near surface than uniform
    Xc_int[:, 2:3] = z_norm
    Xc_int.requires_grad_(True)

    phys_pde = PhysicsLoss2D.residual_pde_only(model, Xc_int)

    # ✅ dt for BC: required to convert rain feature -> actual flux
    if dt_sec_batch is None:
        # fallback if you forgot to pass dt: assume 10 min
        dt_bc = torch.full((Xc_bc.shape[0], 1), 600.0, device=device, dtype=Xc_bc.dtype)
    else:
        dt_bc = dt_sec_batch.to(device).to(Xc_bc.dtype).view(-1, 1)

    phys_bc  = PhysicsLoss2D.residual_bc_surface(model, Xc_bc, dt_bc)

    phys_loss = phys_pde + 1.0 * phys_bc

    # ------------------------
    # Gradients for regularizers
    # ------------------------
    grads_all = torch.autograd.grad(
        outputs=Y_pred,
        inputs=X_batch,
        grad_outputs=torch.ones_like(Y_pred),
        create_graph=True,
        retain_graph=True,
    )[0]  # (N,20)

    # dθ/dz in physical meters: z_norm -> z_m
    grad_theta_znorm = grads_all[:, 2:3]
    Hnorm = float(THICKNESS_TRAIN_NORM)
    grad_theta_z_m = grad_theta_znorm / Hnorm

    grad_loss = torch.mean(grad_theta_z_m ** 2)         # smoothness
    mono_loss = torch.mean(F.relu(grad_theta_z_m))      # discourage θ increasing with depth

    # rainfall-response consistency near surface:
    z_norm_batch = X_batch[:, 2:3]
    rain_col = X_batch[:, 4:5]
    dtheta_dt_hours = grads_all[:, 0:1]  # derivative wrt HOURS input

    z_surface_m = 0.05
    z_surface_norm = z_surface_m / float(THICKNESS_TRAIN_NORM)
    mask_rain_surface = ((rain_col > 1e-6) & (z_norm_batch <= z_surface_norm))

    if mask_rain_surface.any():
        dtheta_dt_rain = dtheta_dt_hours[mask_rain_surface]
        rain_response_loss = torch.mean(F.relu(-dtheta_dt_rain))
    else:
        rain_response_loss = torch.tensor(0.0, device=device)

    # ------------------------
    # Initial condition loss (earliest time in batch)
    # ------------------------
    t_hr = X_batch[:, 0:1]   # hours
    dt_hr = 600.0 / 3600.0   # 10 min in hours
    ic_mask = (t_hr <= dt_hr + 1e-6)

    if ic_mask.any():
        ic_loss = F.mse_loss(Y_pred[ic_mask], Y_batch[ic_mask])
    else:
        ic_loss = torch.tensor(0.0, device=device)

    # ------------------------
    # Penalty LR term
    # ------------------------
    penalty_term = torch.tensor(0.0, device=device)
    if hasattr(model, "penalty_loss") and callable(getattr(model, "penalty_loss")):
        penalty_term = model.penalty_loss()

    alpha_penalty = 1e-5

    # extra rain-weighted MSE
    rain_mask = (rain_col > 1e-6).squeeze(1)
    if rain_mask.any():
        mse_rain = nn.MSELoss()(Y_pred[rain_mask], Y_batch[rain_mask])
    else:
        mse_rain = torch.tensor(0.0, device=device)

    # ------------------------
    # Weight schedule (TUNED: delay physics, prioritize data fitting)
    # ------------------------
    # Phase 1 (epoch < 600): Pure data fitting - NO physics
    # Phase 2 (600 <= epoch < 1000): Slowly introduce physics
    # Phase 3 (epoch >= 1000): Small constant physics weight
    if epoch < 600:
        w_phys = 0.0
    elif epoch < 1000:
        w_phys = 0.005 * (epoch - 600) / 400.0  # ramp from 0 to 0.005
    else:
        w_phys = 0.005  # cap at very small value

    # ------------------------
    # Depth-profile loss: penalize if model ignores depth variation
    # ------------------------
    z_norm_batch = X_batch[:, 2:3]

    # Group by unique (t, x, soil) and compute variance of predictions across depths
    depth_var_pred = torch.var(Y_pred)
    depth_var_obs = torch.var(Y_batch)

    # If observed data has depth variation but prediction doesn't, penalize
    depth_profile_loss = torch.tensor(0.0, device=device)
    if depth_var_obs > 1e-4:
        depth_profile_loss = F.relu(depth_var_obs - depth_var_pred) * 10.0
    else:
        depth_profile_loss = torch.tensor(0.0, device=device)

    theta0_in = X_batch[:, 19:20]  # feature 19
    t_hr = X_batch[:, 0:1]
    t0 = torch.min(t_hr)
    ic_mask = torch.abs(t_hr - t0) <= (600.0/3600.0 + 1e-6)

    theta0_feat_loss = torch.tensor(0.0, device=device)
    if ic_mask.any():
        theta0_feat_loss = F.mse_loss(Y_pred[ic_mask], theta0_in[ic_mask])

    # Recommended weights (TUNED for better deep layer fitting)
    w_grad = 1e-7  # very small - avoid smoothing
    w_mono = 0
    w_ic   = 0.5
    w_depth = 1.0
    w_depth_bin = 2.0  # per-depth-bin loss weight (increased for slope-deep)

    total_loss = (
        1.0 * mse           # depth-weighted MSE
        + 0.3 * mae          # depth-weighted MAE
        + 0.5 * mse_rain
        + w_depth_bin * depth_bin_loss  # emphasize deep layer
        + wetdry_loss       # NEW: wet/dry classification loss
        + w_phys * phys_loss
        + w_ic * ic_loss
        + w_grad * grad_loss
        + w_mono * mono_loss
        + 1e-3 * rain_response_loss
        + alpha_penalty * penalty_term
        + w_depth * depth_profile_loss
        + 0.0 * theta0_feat_loss
        + 0.0 * profile_flatness_loss
    )

    loss_dict = {
        "total":    float(total_loss.item()),
        "mse":      float(mse.item()),
        "mae":      float(mae.item()),
        "mse_deep": float(mse_deep.item()),  # track deep layer error
        "mse_slope_deep": float(mse_slope_deep.item()),  # track slope+deep error
        "wetdry": float(wetdry_loss.item()),  # wet/dry classification loss
        "phys":     float(phys_loss.item()),
        "grad":     float(grad_loss.item()),
        "mono":     float(mono_loss.item()),
        "ic":       float(ic_loss.item()),
        "penalty":  float(penalty_term.item()),
        "w_phys":   float(w_phys),
        "w_grad":   float(w_grad),
        "w_mono":   float(w_mono),
        "w_ic":     float(w_ic),
        "alpha_pen": float(alpha_penalty),
        "mse_rain": float(mse_rain.item()),
        "rain_resp": float(rain_response_loss.item()),
        "depth_prof": float(depth_profile_loss.item()),
    }
    return total_loss, loss_dict

# ==========================================================
# Groundwater level rise (accepts scalar θs or per-pixel θs map)
# ==========================================================
class GroundwaterCalculator:
    def __init__(self, soil_thickness: Optional[float] = None,
                 soil_thickness_map: Optional[np.ndarray] = None):
        self.scalar = soil_thickness
        self.map = soil_thickness_map

    def _layer_dz(self, depths: np.ndarray, H: int, W: int) -> np.ndarray:
        dz_vec = np.diff(np.concatenate([[0.0], depths]))  # (nZ,)
        if self.map is not None:
            # scale to actual local thickness; scale shape [H, W]
            scale = (self.map.astype(np.float32) /
                     max(float(depths[-1]), 1e-6))          # [H, W]
            dz = dz_vec[:, None, None] * scale[None, :, :]  # [nZ, H, W]
        else:
            scale = float(self.scalar) / max(float(depths[-1]), 1e-6)
            dz = dz_vec[:, None, None] * scale               # [nZ, 1, 1]
            dz = np.broadcast_to(dz, (dz_vec.shape[0], H, W)).astype(np.float32)
        return dz.astype(np.float32)

    def calculate(self, theta: np.ndarray, depths: np.ndarray,
                  theta_s: Union[float, np.ndarray]) -> np.ndarray:
        """
        theta: [nT, nZ, H, W]
        theta_s: scalar or [H,W] map
        Returns gwl: [nT, H, W] in meters (equivalent water column height).
        """
        nT, nZ, H, W = theta.shape
        out = np.zeros((nT, H, W), dtype=np.float32)
        dz = self._layer_dz(depths, H, W)  # [nZ, H, W]

        if np.isscalar(theta_s):
            sat_thr = 0.95 * float(theta_s)
            sat_map = None
        else:
            ts = np.asarray(theta_s, dtype=np.float32)
            sat_map = 0.95 * ts  # [H, W]

        for t in range(nT):
            prof = theta[t]  # [nZ, H, W]

            if sat_map is None:
                saturated = prof >= sat_thr
            else:
                saturated = prof >= sat_map[None, :, :]

            any_sat = saturated.any(axis=0)  # [H, W]
            deepest = np.where(any_sat, saturated[::-1].argmax(axis=0), 0)
            deepest = (nZ - 1) - deepest

            mask_bottom_sat = (any_sat) & (deepest == (nZ - 1))
            if mask_bottom_sat.any():
                if sat_map is None:
                    theta_s_use = float(theta_s)
                    excess = np.maximum(prof - theta_s_use, 0.0)  # [nZ,H,W]
                    denom = theta_s_use
                else:
                    theta_s_use = sat_map / 0.95                  # ≈ θs
                    excess = np.maximum(prof - theta_s_use[None, :, :], 0.0)
                    denom = theta_s_use + 1e-6

                water_height = (excess * dz).sum(axis=0) / denom  # [H, W]
                out[t][mask_bottom_sat] = water_height[mask_bottom_sat]

        return out

def parse_slope_from_filename(fname: str, default: float = 15.0) -> float:
    """
    Extract slope angle in degrees from filename.
    Examples:
      GR_s10_90mmh.xlsx  -> 10
      CP_S30_r50.xlsx    -> 30
      Red_s5.xlsx        -> 5

    It looks for patterns like 's10', 'S30', etc.
    If nothing is found, it returns the default value.
    """
    base = os.path.splitext(os.path.basename(fname))[0]  # remove folder & .xlsx

    # Pattern 1: 's10', 'S15', 's30' etc.
    m = re.search(r'[sS](\d+)', base)
    if m:
        try:
            return float(m.group(1))
        except:
            pass

    # Optional extra: pattern like '10deg', '30deg'
    m = re.search(r'(\d+)\s*deg', base, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except:
            pass

    # Fallback
    return default

# ==========================================================
# Trainer (ANN data loss + physics loss) - TUNED
# ==========================================================
class PINNTrainer:
    def __init__(self, model:EnhancedPINN, dl:DataLoader):
        self.model = model.to(device)
        self.dl = dl
        # TUNED: Higher initial LR, less weight decay for better data fitting
        self.opt = optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=5e-5)
        # TUNED: Longer warm restart period for stable convergence
        self.sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=200, T_mult=2, eta_min=1e-6
        )

    def _prep_train(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build training X (N,18), Y (N,1), and DT (N,1) from dl.excel_data.

        - time_sec (Excel) -> t_hr (feature 0)
        - depth_m -> z_norm (feature 2) using THICKNESS_TRAIN_NORM
        - rainfall feature (feature 4) stays as mm/10min-equivalent
          (your DataLoader should already convert/aggregate to mm/10min-equivalent)
        - DT returns the REAL dt per moisture interval (seconds), repeated for each depth sample
          so Physics BC can convert rain feature -> actual flux.
        """
        Xs: List[List[float]] = []
        Ys: List[List[float]] = []
        DTs: List[List[float]] = []

        soil_thickness_m_train = float(THICKNESS_TRAIN_NORM)
        denom = max(float(soil_thickness_m_train), 1e-6)

        for fname, data in self.dl.excel_data.items():
            # Required series
            T_sec   = np.asarray(data['time'], dtype=np.float32)         # (T,) seconds
            D_m     = np.asarray(data['depths'], dtype=np.float32)       # (nZ,) meters
            M_mat   = np.asarray(data['moisture'], dtype=np.float32)     # (nZ, T)
            R_10eq  = np.asarray(data['rainfall'], dtype=np.float32)     # (T,) mm/10min-equivalent (NN feature)

            # ✅ REAL dt per moisture interval (seconds)
            dt_arr = data.get('dt_sec', None)
            if dt_arr is None:
                dt_arr = np.full_like(T_sec, 600.0, dtype=np.float32)    # fallback: 10 min
            else:
                dt_arr = np.asarray(dt_arr, dtype=np.float32)
                if len(dt_arr) != len(T_sec):
                    raise ValueError(f"{fname}: dt_sec length {len(dt_arr)} != time length {len(T_sec)}")

            # Optional time series (already filled in DataLoader)
            met     = data['met_series']     # dict of (T,)
            lulc    = data['lulc_series']    # dict of (T,)
            slope_T = np.asarray(data['slope_deg'], dtype=np.float32)    # (T,)
            aspect_T = np.asarray(data.get('aspect_deg', np.zeros_like(T_sec)), dtype=np.float32)

            # Soil one-hot
            w_soil = np.asarray(data['soil_one_hot'], dtype=np.float32)  # (4,)
            soil_w_T = np.tile(w_soil[None, :], (len(T_sec), 1)).astype(np.float32)

            # Feature transforms
            T_hr = T_sec / 3600.0                                     # (T,)
            D_norm = D_m / denom                                       # (nZ,)

            # Safety checks
            T = len(T_sec)
            nZ = len(D_m)
            if M_mat.shape != (nZ, T):
                raise ValueError(f"{fname}: moisture shape {M_mat.shape} != (nZ,T)=({nZ},{T})")

            # Soil one-hot (MUST be 5 = [CP,GR,SD,DC,SC])
            w_soil = np.asarray(data['soil_one_hot'], dtype=np.float32).reshape(-1)
            if w_soil.size != 5:
                raise ValueError(f"{fname}: soil_one_hot must have 5 elements [CP,GR,SD,DC,SC], got {w_soil.size}")
            soil_w_T = np.tile(w_soil[None, :], (len(T_sec), 1)).astype(np.float32)

            wCP, wGR, wSD, wDC, wSC = [float(v) for v in w_soil]

            # theta0 per depth = moisture at first time step
            theta0_by_depth = M_mat[:, 0].astype(np.float32)  # (nZ,)

            for ti in range(T):
                t_hr = float(T_hr[ti])
                slope_deg  = float(slope_T[ti])
                rain10eq   = float(R_10eq[ti])
                dt_sec     = float(dt_arr[ti])
                aspect_deg = float(aspect_T[ti])

                tempC = float(met['temp_C'][ti])
                RH    = float(met['RH_pct'][ti])
                wind  = float(met['wind_ms'][ti])
                wdir  = float(met['wind_dir_deg'][ti])

                FVC_t = float(lulc['FVC'][ti])
                ISA_t = float(lulc['ISA'][ti])

                runoff10eq = RunoffCalculator.runoff_generation_mm10(
                    rain_mm10=np.array([rain10eq], dtype=np.float32),
                    soil_w=soil_w_T[ti:ti+1, :],   # (1,5)
                    FVC=np.array([FVC_t], dtype=np.float32),
                    ISA=np.array([ISA_t], dtype=np.float32),
                    slope_deg=np.array([slope_deg], dtype=np.float32),
                    cap_scale=1.0
                )[0]

                for di in range(nZ):
                    theta_obs = float(M_mat[di, ti])
                    z_norm = float(D_norm[di])

                    theta0 = float(theta0_by_depth[di])  # ✅ per-depth IC feature

                    # ✅ EXACTLY 20 columns, matching INPUT_FEATURES (0..19)
                    Xs.append([
                        t_hr,                    # 0
                        0.0,                     # 1
                        z_norm,                  # 2
                        slope_deg,               # 3
                        rain10eq,                # 4
                        float(runoff10eq),       # 5
                        FVC_t,                   # 6
                        ISA_t,                   # 7
                        wCP, wGR, wSD, wDC, wSC, # 8..12 (5 soils)
                        tempC,                   # 13
                        RH,                      # 14
                        wind,                    # 15
                        wdir,                    # 16
                        soil_thickness_m_train,  # 17
                        aspect_deg,              # 18
                        theta0                   # 19 ✅
                    ])
                    Ys.append([theta_obs])
                    DTs.append([dt_sec])

        X = torch.tensor(Xs, dtype=torch.float32, device=device)
        Y = torch.tensor(Ys, dtype=torch.float32, device=device)
        DT = torch.tensor(DTs, dtype=torch.float32, device=device)

        print("Training samples:", X.shape[0])
        print("Input shape:", X.shape, "(N,18)")
        print("Output shape:", Y.shape, "(N,1)")
        print("DT shape:", DT.shape, "(N,1)")
        print("✅ time feature: hours | z feature: z/thickness_train_norm =", soil_thickness_m_train)
        print("✅ dt_sec stats [min/median/max]:",
              float(torch.min(DT)), float(torch.median(DT)), float(torch.max(DT)))

        return X, Y, DT

    def train(self, epochs:int = 1500) -> List[float]:
        X, Y, DT = self._prep_train()   # ✅ now 3 outputs

        best = float('inf')
        best_mse = float('inf')  # TUNED: track MSE separately
        patience = 0
        max_patience = 1000  # TUNED: longer patience
        hist: List[float] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.opt.zero_grad()

            loss, loss_info = pinn_loss_block(
                self.model,
                X_batch=X,
                Y_batch=Y,
                epoch=epoch,
                dt_sec_batch=DT,      # ✅ pass DT into BC
                collocation_X=None,
                device=device
            )

            loss.backward()
            # TUNED: gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            self.sch.step(epoch + 0.0)

            hist.append(loss.item())

            if epoch % 25 == 0:
                print(
                    f"Epoch {epoch:5d} | "
                    f"Loss={loss_info['total']:.6f} "
                    f"(MSE={loss_info['mse']:.6f}, "
                    f"MSE_deep={loss_info.get('mse_deep', 0):.6f}, "
                    f"Phys={loss_info['phys']:.2e}, "
                    f"wP={loss_info['w_phys']:.1e})"
                )

            # --- TUNED: best model save based on MSE (data fit) after longer burn-in ---
            start_best = 300  # Start earlier since we prioritize data fitting
            if epoch >= start_best:
                # Save based on MSE, not total loss (prioritize data fitting)
                current_mse = loss_info['mse']
                if current_mse < best_mse - 1e-7:
                    best_mse = current_mse
                    best = loss.item()
                    patience = 0
                    torch.save(
                        {'model_state_dict': self.model.state_dict(),
                        'epoch': epoch,
                        'loss': best,
                        'mse': best_mse},
                        os.path.join(self.dl.project_dir, 'best_pinn_model.pth')
                    )
                else:
                    patience += 1

                if patience >= max_patience:
                    print("Early stopping.")
                    break

        ckpt_path = os.path.join(self.dl.project_dir, 'best_pinn_model.pth')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            mse_str = f", MSE={ckpt.get('mse', 'N/A')}" if 'mse' in ckpt else ""
            print(f"Training done. Best loss {ckpt['loss']:.8f}{mse_str} at epoch {ckpt['epoch']}")
        else:
            print("Training done. (No best_pinn_model.pth saved; using last epoch weights.)")

        return hist

# ==========================================================
# Predictor (ALL inputs from rasters; soil weights map or scalar) – vectorized
# ==========================================================

def _read_excel_rowwise_robust(excel_path: str):
    df = pd.read_excel(excel_path, header=None)

    def row_vals(r):
        vals = pd.to_numeric(r[1:], errors='coerce').to_numpy(dtype='float64')
        return vals

    time_vals = None
    rain_10 = None
    rain_h  = None
    met = {'temp_C': None, 'RH_pct': None, 'wind_ms': None, 'wind_dir_deg': None}
    lulc = {'FVC': None, 'ISA': None}
    aspect_deg = None

    depths_cm = []
    moist_rows = []

    for r in range(len(df)):
        label = df.iloc[r, 0]
        if label is None or (isinstance(label, float) and np.isnan(label)):
            continue

        # string label rows
        if isinstance(label, str):
            k = label.strip().lower()

            if k in ("time_sec", "time", "time_s") or ("time" in k):
                time_vals = row_vals(df.iloc[r, :]); continue

            if k == "rain_mm_10min":
                rain_10 = row_vals(df.iloc[r, :]); continue
            if k == "rain_mm_h":
                rain_h = row_vals(df.iloc[r, :]); continue

            if k == "temp_c":
                met["temp_C"] = row_vals(df.iloc[r, :]); continue
            if k == "rh_pct":
                met["RH_pct"] = row_vals(df.iloc[r, :]); continue
            if k == "wind_ms":
                met["wind_ms"] = row_vals(df.iloc[r, :]); continue
            if k == "wind_dir_deg":
                met["wind_dir_deg"] = row_vals(df.iloc[r, :]); continue

            if k == "fvc":
                lulc["FVC"] = row_vals(df.iloc[r, :]); continue
            if k == "isa":
                lulc["ISA"] = row_vals(df.iloc[r, :]); continue

            if k in ("aspect", "aspect_deg"):
                aspect_deg = row_vals(df.iloc[r, :]); continue

            # depth label like depth_cm=5
            if k.startswith("depth_cm="):
                try:
                    dcm = float(k.split("=")[1])
                    vals = row_vals(df.iloc[r, :])
                    if not np.isnan(vals).all():
                        depths_cm.append(dcm)
                        moist_rows.append(vals)
                    continue
                except:
                    pass

        # numeric depth label like 5, 15, 30
        try:
            dcm = float(str(label).strip())
            vals = row_vals(df.iloc[r, :])
            if not np.isnan(vals).all():
                depths_cm.append(dcm)
                moist_rows.append(vals)
        except:
            pass

    if time_vals is None:
        raise ValueError(f"Missing time row in {excel_path}")

    T = len(time_vals)

    # rainfall -> mm/10min-equivalent
    if rain_10 is None and rain_h is None:
        raise ValueError(f"Missing rainfall row in {excel_path}")

    # use time to infer dt
    t_moist = np.asarray(time_vals, dtype='float64')
    if t_moist.size < 2:
        dt0 = 600.0
    else:
        dt0 = float(np.nanmedian(np.diff(t_moist)))
        if not np.isfinite(dt0) or dt0 <= 0:
            dt0 = 600.0

    if rain_10 is not None:
        rain_10 = np.asarray(rain_10, dtype='float64')
    else:
        rain_h = np.asarray(rain_h, dtype='float64')
        if len(rain_h) == len(t_moist):
            mm_step = rain_h * (dt0 / 3600.0)   # mm over dt
            rain_10 = mm_step * (600.0 / dt0)   # mm/10min-equivalent
        else:
            rain_10 = rain_h / 6.0


    def fill_or_check(arr, default):
        if arr is None:
            return np.full(T, default, dtype='float64')
        arr = np.asarray(arr, dtype='float64')
        if len(arr) != T:
            raise ValueError(f"Series length mismatch: {len(arr)} vs T={T}")
        return arr

    def clip01(a):
        return np.clip(np.asarray(a, dtype='float64'), 0.0, 1.0)

    met = {
        'temp_C':       fill_or_check(met['temp_C'], 25.0),
        'RH_pct':       fill_or_check(met['RH_pct'], 70.0),
        'wind_ms':      fill_or_check(met['wind_ms'], 1.0),
        'wind_dir_deg': fill_or_check(met['wind_dir_deg'], 0.0),
    }
    lulc = {
        'FVC': clip01(fill_or_check(lulc['FVC'], 0.5)),
        'ISA': clip01(fill_or_check(lulc['ISA'], 0.2)),
    }
    aspect_deg = fill_or_check(aspect_deg, 0.0)

    if len(depths_cm) == 0:
        depths_cm_obs = np.array([], dtype='float64')
        obs = np.array([[]], dtype='float64')
    else:
        depths_cm = np.asarray(depths_cm, dtype='float64')
        obs = np.vstack(moist_rows).astype('float64')[:, :T]
        order = np.argsort(depths_cm)
        depths_cm_obs = depths_cm[order]
        obs = obs[order, :]

    return {
        'time_s': np.asarray(time_vals, dtype='float64'),
        'rain_mm_10min': np.asarray(rain_10, dtype='float64'),
        'met': met,
        'lulc': lulc,
        'aspect_deg': np.asarray(aspect_deg, dtype='float64'),
        'depths_cm_obs': depths_cm_obs,
        'water_content_obs': obs
    }

def _build_uniform_forcing_from_excel(excel_path: str) -> dict:
    """
    Read an Excel file that contains only *boundary / forcing* time series,
    NOT monitoring depths. It must have (row labels in col A):

      - time or time_sec
      - rain_mm_10min or rain_mm_h
      - temp_C
      - RH_pct
      - wind_ms
      - wind_dir_deg
      - fvc
      - isa
      - (optional) aspect_deg or aspect

    Returns dict with time_s and per-variable 1D arrays.
    """
    # ✅ use the argument, not a hard-coded file
    series = _read_excel_rowwise_robust(excel_path)

    time_s = series['time_s'].astype(np.float32)
    rain_10 = series['rain_mm_10min'].astype(np.float32)  # already mm/10min
    met = series['met']
    lulc = series['lulc']
    aspect_deg = series.get('aspect_deg', np.zeros_like(time_s))

    forcing = {
        'time_s':      time_s,
        'rain_mm_10':  rain_10,               # mm/10min (for model input)
        'rain_mm_h':   rain_10 * 6.0,         # mm/h  (for runoff, etc.)
        'temp_C':      met['temp_C'].astype(np.float32),
        'RH_pct':      met['RH_pct'].astype(np.float32),
        'wind_ms':     met['wind_ms'].astype(np.float32),
        'wind_dir_deg':met['wind_dir_deg'].astype(np.float32),
        'FVC':         lulc['FVC'].astype(np.float32),
        'ISA':         lulc['ISA'].astype(np.float32),
        'aspect_deg':  aspect_deg.astype(np.float32),
    }
    return forcing

class PINNPredictor:
    def __init__(self, model, dl, device=device):
        self.model = model
        self.dl = dl
        self.device = device

        # Always available
        self.slope = dl.slope_data
        self.thick = dl.soil_thickness_arr

        # Optional (may be None)
        self.aspect = getattr(dl, "aspect_data", None)

        # LULC is stored here (static rasters) if available:
        # dl.landuse_arrays['FVC'], dl.landuse_arrays['ISA']
        self.landuse = getattr(dl, "landuse_arrays", None)

        # ✅ DEM-aware runoff router (optional DEM)
        self.runoff = RunoffCalculator(dem=getattr(dl, "dem_data", None))

        # Uniform forcing (optional) — initialize to avoid AttributeError
        self.uniform_forcing = None

    # ---------- PUBLIC: set uniform forcing from Excel ----------
    def load_uniform_forcing_from_excel(self, excel_path: str):
        """
        Load a time-varying but spatially-uniform forcing (rain + met + FVC/ISA + aspect)
        from an Excel file. After calling this, prediction will prioritize:
            uniform forcing (Excel)  ->  rasters  ->  error/default.
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Forcing Excel not found: {excel_path}")
        self.uniform_forcing = _build_uniform_forcing_from_excel(excel_path)
        print(f"Loaded uniform forcing from: {excel_path}")
        print("Forcing time range [s]:", self.uniform_forcing['time_s'][0],
              "→", self.uniform_forcing['time_s'][-1])

    # ---------- helpers: nearest time index in a 1D time series ----------
    @staticmethod
    def _closest_idx(times: np.ndarray, t: float) -> int:
        return int(np.argmin(np.abs(times - float(t))))

    # ---------- helpers: rasters (already in DataLoader) ----------
    @staticmethod
    def _closest_key(d: dict, t: float) -> Optional[int]:
        if not d:
            return None
        return min(d.keys(), key=lambda k: abs(k - t))

    # ---------- rainfall fields (returns mm/10min for model input) ----------
    def _get_rain_mm10_field(self, t: float, H: int, W: int) -> np.ndarray:
        """
        Returns rainfall in mm/10min (because the model was trained with mm/10min):
        - If uniform_forcing is set → use its time series.
        - Else if rainfall_maps exist → convert rasters mm/h → mm/10min.
        - Else → ERROR (must provide rainfall as Excel or rasters).
        """
        # 1) uniform forcing from Excel
        if self.uniform_forcing is not None:
            tf = self.uniform_forcing['time_s']
            idx = self._closest_idx(tf, t)
            r10 = float(self.uniform_forcing['rain_mm_10'][idx])  # mm/10min
            return np.full((H, W), r10, dtype=np.float32)

        # 2) rainfall rasters (mm/h) from /rain folder
        if self.dl.rainfall_maps:
            tt = self._closest_key(self.dl.rainfall_maps, t)
            mmh = self.dl.rainfall_maps[tt].astype(np.float32)  # mm/h
            return mmh / 6.0  # → mm/10min for model

        # 3) nothing provided → error
        raise RuntimeError(
            "No rainfall provided for prediction. "
            "You must either:\n"
            "  - call predictor.load_uniform_forcing_from_excel(excel_path), OR\n"
            "  - provide rain/rain_tXXXX.tif rasters."
        )

    # ---------- rainfall for runoff (mm/h) ----------
    def _get_rain_mm_h_for_runoff(self, t: float, H: int, W: int) -> np.ndarray:
        """
        Returns rainfall in mm/h for the runoff calculator.
        Priority: uniform forcing → rainfall rasters → error.
        """
        if self.uniform_forcing is not None:
            tf = self.uniform_forcing['time_s']
            idx = self._closest_idx(tf, t)
            mmh = float(self.uniform_forcing['rain_mm_h'][idx])
            return np.full((H, W), mmh, dtype=np.float32)

        if self.dl.rainfall_maps:
            tt = self._closest_key(self.dl.rainfall_maps, t)
            return self.dl.rainfall_maps[tt].astype(np.float32)

        raise RuntimeError(
            "No rainfall provided (Excel or rasters) to compute runoff."
        )

    # ---------- meteorological fields ----------
    def _get_met_field(self, name: str, t: float, H: int, W: int) -> np.ndarray:
        """
        name in {'temp_C','RH_pct','wind_ms','wind_dir_deg'}.
        Priority: uniform forcing → met rasters → error.
        """
        # 1) uniform forcing from Excel
        if self.uniform_forcing is not None:
            tf = self.uniform_forcing['time_s']
            idx = self._closest_idx(tf, t)
            val = float(self.uniform_forcing[name][idx])
            return np.full((H, W), val, dtype=np.float32)

        # 2) raster stacks from dl.met_maps
        stack = self.dl.met_maps.get(name, {})
        if stack:
            tt = self._closest_key(stack, t)
            return stack[tt].astype(np.float32)

        # 3) nothing provided → error
        raise RuntimeError(
            f"No {name} provided for prediction. "
            "Provide it as:\n"
            "  - Excel forcing (predictor.load_uniform_forcing_from_excel), or\n"
            "  - met/{temp,rh,wind_ms,wind_dir}_tXXXX.tif rasters."
        )

    # ---------- LULC fields ----------
    def _get_lulc_field(self, name: str, t: float, H: int, W: int) -> np.ndarray:
        """
        name in {'FVC','ISA'}.
        Priority: uniform forcing → static LULC rasters → error.
        (We treat FVC/ISA as time-varying in Excel, but static in rasters.)
        """
        # 1) uniform forcing from Excel
        if self.uniform_forcing is not None:
            tf = self.uniform_forcing['time_s']
            idx = self._closest_idx(tf, t)
            val = float(self.uniform_forcing[name][idx])
            return np.full((H, W), val, dtype=np.float32)

        # 2) static rasters from DataLoader.landuse_arrays
        if self.dl.landuse_arrays is not None and name in self.dl.landuse_arrays:
            return self.dl.landuse_arrays[name].astype(np.float32)

        raise RuntimeError(
            f"No {name} provided for prediction. "
            "Provide it as:\n"
            "  - Excel forcing (time-varying uniform), or\n"
            "  - LULC rasters in lulc/fvc.tif and lulc/isa.tif."
        )

    # ---------- aspect field ----------
    def _get_aspect_field(self, t: float, H: int, W: int) -> np.ndarray:
        """
        Aspect in degrees (azimuth). Priority:
          - uniform forcing Excel (time-varying uniform),
          - Aspect.tif (static),
          - default 0° if nothing else.
        """
        # 1) uniform forcing from Excel
        if self.uniform_forcing is not None and 'aspect_deg' in self.uniform_forcing:
            tf = self.uniform_forcing['time_s']
            idx = self._closest_idx(tf, t)
            val = float(self.uniform_forcing['aspect_deg'][idx])
            return np.full((H, W), val, dtype=np.float32)

        # 2) static aspect raster
        if getattr(self.dl, 'aspect_data', None) is not None:
            return self.dl.aspect_data.astype(np.float32)

        # 3) default
        return np.zeros((H, W), dtype=np.float32)

    # ---------- soil weights (from soil_type.tif or user) ----------
    def _make_soil_mask(self, soil_type_arg: Optional[Union[str,int,float,List[float],Tuple[float,...]]]) -> np.ndarray:
        H, W = self.dl.slope_data.shape
        if self.dl.soil_type_oh is not None:
            return self.dl.soil_type_oh.astype(np.float32)

        if soil_type_arg is None:
            raise ValueError(
                "No soil_type.tif found and no soil_type_arg provided. "
                "Provide it as 'CP'|'GR'|'SD'|'DC'|'SC', int code 0..4, or a 5-element weight vector."
                "or a 5-element weight vector."
            )

        if isinstance(soil_type_arg, (list, tuple)) and len(soil_type_arg) == 5:
            w = np.asarray(soil_type_arg, dtype=np.float32)
            s = max(w.sum(), 1e-6)
            w = w / s
            return np.tile(w[None, None, :], (H, W, 1)).astype(np.float32)

        if isinstance(soil_type_arg, (int, float)):
            code = int(soil_type_arg)
            if code not in (0, 1, 2, 3, 4):
                raise ValueError("Numeric soil_type_arg must be 0..4 (0=CP,1=GR,2=SD,3=DC,4=SC).")
            w = np.zeros((5,), dtype=np.float32)
            w[code] = 1.0
            return np.tile(w[None, None, :], (H, W, 1)).astype(np.float32)

        tag = str(soil_type_arg).upper()
        if tag not in SoilParameters.ORDER:
            raise ValueError("soil_type_arg must be one of 'CP','GR','SD','DC','SC'.")
        w = np.zeros((5,), dtype=np.float32)
        w[SoilParameters.ORDER.index(tag)] = 1.0
        return np.tile(w[None, None, :], (H, W, 1)).astype(np.float32)

    # ---------- MAIN PREDICTION FUNCTION ----------
    def predict_theta(
        self,
        time_range: np.ndarray,
        depth_range: np.ndarray,
        soil_type_arg=None,
        batch_pixels: int = 20000
    ) -> np.ndarray:
        """
        Predict θ(t,z,x) over the grid.

        API expectation:
          - time_range is in SECONDS (recommended). If values look like HOURS, it will auto-convert.
          - depth_range is in METERS (physical depth).
          - model inputs: t_hr (hours), z_norm = z_m / THICKNESS_TRAIN_NORM

        Requires:
          - Slope.tif
          - soil/soil_thickness.tif
          - forcings via either uniform Excel forcing OR raster stacks
        """
        if self.dl.slope_data is None:
            raise ValueError("Slope grid missing.")
        if self.dl.soil_thickness_arr is None:
            raise ValueError("Soil thickness map missing (soil/soil_thickness.tif).")

        time_range = np.asarray(time_range, dtype=np.float32)
        depth_range = np.asarray(depth_range, dtype=np.float32)

        # ✅ Safety: if times are small (likely hours), convert to seconds
        if np.nanmax(time_range) < 1e3:
            time_range = time_range * 3600.0

        slope = self.dl.slope_data.astype(np.float32)
        thick = self.dl.soil_thickness_arr.astype(np.float32)

        H, W = slope.shape
        nT, nZ = len(time_range), len(depth_range)
        HW = H * W

        soil_w = self._make_soil_mask(soil_type_arg)  # (H,W,5)

        # --- theta0 baseline per pixel: use blended theta_r as initial condition proxy ---
        theta_r_vec = np.array([
            SoilParameters.CP['theta_r'],
            SoilParameters.GR['theta_r'],
            SoilParameters.SD['theta_r'],
            SoilParameters.DC['theta_r'],
            SoilParameters.SC['theta_r'],
        ], dtype=np.float32)  # (5,)

        # soil_w is (H, W, 5)
        theta0_base = (soil_w * theta_r_vec[None, None, :]).sum(axis=2).astype(np.float32)  # (H, W)
        theta0_flat = theta0_base.reshape(HW, 1)  # (HW, 1)

        # x coordinate (meters) based on pixel size
        px = getattr(self.dl, 'pixel_size_x', 1.0)
        xcoord = (np.arange(W, dtype=np.float32)[None, :] * float(px)).repeat(H, axis=0)

        x_flat     = xcoord.ravel()
        thick_flat = thick.ravel()
        soil_flat  = soil_w.reshape(HW, 5)

        self.model.eval()
        theta = np.zeros((nT, nZ, H, W), dtype=np.float32)

        with torch.no_grad():
            for ti, t_s in enumerate(tqdm(time_range, desc="Predict")):
                # forcings (model wants rain in mm/10min)
                rain10 = self._get_rain_mm10_field(float(t_s), H, W)
                FVC    = self._get_lulc_field('FVC', float(t_s), H, W)
                ISA    = self._get_lulc_field('ISA', float(t_s), H, W)
                aspect = self._get_aspect_field(float(t_s), H, W)

                temp = self._get_met_field('temp_C',       float(t_s), H, W)
                rh   = self._get_met_field('RH_pct',       float(t_s), H, W)
                ws   = self._get_met_field('wind_ms',      float(t_s), H, W)
                wd   = self._get_met_field('wind_dir_deg', float(t_s), H, W)

                # runoff generation (mm/10min) + optional routing
                runoff_gen = RunoffCalculator.runoff_generation_mm10(
                    rain_mm10=rain10,
                    soil_w=soil_w,
                    FVC=FVC,
                    ISA=ISA,
                    slope_deg=slope
                )
                runoff = self.runoff.route_runoff_mm10(runoff_gen) if (self.runoff.dem is not None) else runoff_gen

                # base features (flat)
                base_features = np.stack([
                    slope.ravel(),        # 3
                    rain10.ravel(),       # 4
                    runoff.ravel(),       # 5
                    FVC.ravel(),          # 6
                    ISA.ravel(),          # 7
                ], axis=1).astype(np.float32)

                met_features = np.stack([
                    temp.ravel(),         # 13
                    rh.ravel(),           # 14
                    ws.ravel(),           # 15
                    wd.ravel(),           # 16
                ], axis=1).astype(np.float32)

                thick_col = np.full((HW, 1), float(THICKNESS_TRAIN_NORM), dtype=np.float32) # 17
                aspect_flat = aspect.ravel().reshape(HW, 1).astype(np.float32)  # 18

                # time input (hours)
                t_hr = float(t_s) / 3600.0
                t_col = np.full((HW, 1), t_hr, dtype=np.float32)
                x_col = x_flat.reshape(HW, 1).astype(np.float32)

                for zi, z_m in enumerate(depth_range):
                    # ✅ IMPORTANT: model uses z_norm normalized by THICKNESS_TRAIN_NORM (not local thickness)
                    z_norm = float(z_m) / float(THICKNESS_TRAIN_NORM)
                    z_col = np.full((HW, 1), z_norm, dtype=np.float32)

                    feat_all = np.concatenate([
                        t_col, x_col, z_col,         # 0..2
                        base_features,               # 3..7
                        soil_flat,                   # 8..12 (5)
                        met_features,                # 13..16 (4)
                        thick_col,                   # 17
                        aspect_flat,                  # 18
                        theta0_flat                  # 19
                    ], axis=1).astype(np.float32)

                    out_flat = np.empty((HW,), dtype=np.float32)
                    for start in range(0, HW, batch_pixels):
                        end = min(start + batch_pixels, HW)
                        Xb = torch.from_numpy(feat_all[start:end]).to(self.device)
                        out_flat[start:end] = self.model(Xb).squeeze(1).cpu().numpy().astype(np.float32)

                    theta[ti, zi] = out_flat.reshape(H, W)

        return theta

# ==========================================================
# Visualizer & water front utilities
# ==========================================================
class Visualizer:
    def __init__(self, project_dir: str):
        self.outdir = os.path.join(project_dir, "outputs")
        os.makedirs(self.outdir, exist_ok=True)

    # ✅ FIX: add the missing method
    def plot_training_loss(
        self,
        loss_hist,
        fname: str = "training_loss.png",
        save_csv: bool = True,
        csv_name: str = "training_loss.csv",
        use_logy: bool = True
    ):
        """
        Plot training loss history (list/np array/torch tensor) and save to outputs/.
        """
        # Convert to 1D numpy
        if isinstance(loss_hist, torch.Tensor):
            loss = loss_hist.detach().cpu().numpy().astype(np.float64).ravel()
        else:
            loss = np.asarray(loss_hist, dtype=np.float64).ravel()

        if loss.size == 0:
            print("Warning: loss_hist is empty; skip plot_training_loss().")
            return None

        # Optional CSV export
        if save_csv:
            try:
                df = pd.DataFrame({"epoch": np.arange(1, len(loss) + 1), "loss": loss})
                df.to_csv(os.path.join(self.outdir, csv_name), index=False)
            except Exception as e:
                print("Warning: cannot save loss CSV:", e)

        # Plot
        plt.figure(figsize=(8, 5))
        x = np.arange(1, len(loss) + 1)

        if use_logy and np.all(loss > 0):
            plt.semilogy(x, loss)
            plt.ylabel("Loss (log scale)")
        else:
            plt.plot(x, loss)
            plt.ylabel("Loss")

        plt.xlabel("Epoch")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.outdir, fname)
        plt.savefig(outpath, dpi=300)
        plt.close()
        print("Saved:", outpath)
        return outpath

    @staticmethod
    def compute_water_front_depth(theta_t: np.ndarray,
                                  depths: np.ndarray,
                                  theta_thr: float) -> float:
        mask = theta_t >= theta_thr
        if not np.any(mask):
            return 0.0
        idx = np.where(mask)[0][-1]
        return float(depths[idx])

    def plot_water_front_timeseries(
        self,
        theta: np.ndarray,
        depths: np.ndarray,
        theta_r: float,
        theta_s: float,
        time_range: np.ndarray,
        ij: Tuple[int, int],
        depth_mark: float = 0.10,   # meters
        fname: str = "water_front_timeseries.png",
        front_frac: float = 0.90
    ) -> Optional[float]:

        i, j = ij
        nT, nZ, H, W = theta.shape

        depths = np.asarray(depths, dtype=np.float32)
        theta_thr = float(theta_r + front_frac * (theta_s - theta_r))

        front = np.zeros((nT,), dtype=np.float32)
        for ti in range(nT):
            prof = theta[ti, :, i, j]
            front[ti] = self.compute_water_front_depth(prof, depths, theta_thr)

        t_cross = None
        for k in range(1, nT):
            if (front[k-1] < depth_mark) and (front[k] >= depth_mark):
                f0, f1 = front[k-1], front[k]
                t0, t1 = time_range[k-1], time_range[k]
                t_cross = t0 + (depth_mark - f0) * (t1 - t0) / (f1 - f0 + 1e-12)
                break

        plt.figure(figsize=(8, 5))
        plt.plot(time_range, front, label=f"Water-front depth (θ ≥ {theta_thr:.3f})")
        plt.axhline(depth_mark, linestyle='--', label=f"Depth mark = {depth_mark:.2f} m")

        if t_cross is not None:
            plt.axvline(t_cross, linestyle=':', label=f"Crossing at t={t_cross:.0f} s")

        plt.xlabel("Time (s)")
        plt.ylabel("Depth (m)")
        plt.title(f"Water front at pixel (i={i}, j={j})")
        plt.grid(True, alpha=0.3)
        plt.legend()

        outpath = os.path.join(self.outdir, fname)
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()

        return float(t_cross) if t_cross is not None else None

    # ✅ Make it staticmethod (no self) so it works cleanly
    @staticmethod
    def export_excel_template(project_dir: str,
                             filename: str = "PINN_training_template.xlsx",
                             T: int = 13,
                             depths_cm=(10, 30, 60, 100)) -> str:
        """
        Create a row-wise Excel template matching DataLoader.load_excel():
          Col A = labels; Col B.. = values for each time-step
          Required rows: time_sec, rain_mm_10min, temp_C, RH_pct, wind_ms, wind_dir_deg, fvc, isa
          Optional: aspect_deg
          Depth rows: 'depth_cm=<number>'
        """
        import pandas as pd
        import numpy as np
        os.makedirs(project_dir, exist_ok=True)

        dt = 600.0
        time_sec = np.arange(T, dtype=np.float64) * dt

        rain_mm_10min = np.zeros(T, dtype=np.float64)
        rain_mm_10min[3:6] = 25.0 / 6.0  # example 25 mm/h storm for 30 min

        temp_C = np.full(T, 25.0, dtype=np.float64)
        RH_pct = np.full(T, 70.0, dtype=np.float64)
        wind_ms = np.full(T, 1.5, dtype=np.float64)
        wind_dir_deg = np.full(T, 90.0, dtype=np.float64)
        fvc = np.clip(np.linspace(0.4, 0.8, T), 0, 1)
        isa = np.clip(0.3 - 0.02 * np.arange(T), 0, 1)
        aspect_deg = np.full(T, 0.0, dtype=np.float64)

        moist_by_depth = []
        for dcm in depths_cm:
            base = 0.20 + 0.001 * dcm
            theta = np.full(T, base, dtype=np.float64)
            theta[4:7] += 0.05 * np.exp(-dcm / 60.0)
            theta = np.clip(theta, 0.05, 0.45)
            moist_by_depth.append(theta)

        rows = []

        def add_row(label, arr):
            rows.append([label] + list(map(float, arr)))

        add_row('time_sec', time_sec)
        add_row('rain_mm_10min', rain_mm_10min)
        add_row('temp_C', temp_C)
        add_row('RH_pct', RH_pct)
        add_row('wind_ms', wind_ms)
        add_row('wind_dir_deg', wind_dir_deg)
        add_row('fvc', fvc)
        add_row('isa', isa)
        add_row('aspect_deg', aspect_deg)

        for dcm, theta in zip(depths_cm, moist_by_depth):
            add_row(f'depth_cm={dcm}', theta)

        df = pd.DataFrame(rows)
        out_path = os.path.join(project_dir, filename)
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
            df.to_excel(xw, index=False, header=False, sheet_name="training")

        print(f"Excel template written to: {out_path}")
        return out_path

# === === === === === === === === ===
# === PREDICTION FUNCTIONS (POINT) ===
# === === === === === === === === ===

def _parse_soil_slope_from_filename(excel_path: str):
    stem = os.path.splitext(os.path.basename(excel_path))[0].upper()

    # token-based soil detection (same idea as DataLoader)
    tokens = re.split(r'[^A-Z0-9]+', stem)
    tokens = [t for t in tokens if t]

    soil = "GR"  # safe default
    for tag in SoilParameters.ORDER:  # ['CP','GR','SD','DC','SC']
        if tag in tokens:
            soil = tag
            break

    # slope: reuse existing parser
    slope = parse_slope_from_filename(stem, default=0.0)
    return soil, slope


def _soil_onehot(soil_tag: str) -> np.ndarray:
    order = SoilParameters.ORDER  # ['CP','GR','SD','DC','SC']
    w = np.zeros(5, dtype=np.float32)
    if soil_tag in order:
        w[order.index(soil_tag)] = 1.0
    else:
        w[order.index('GR')] = 1.0  # fallback
    return w

def _build_point_features_20(
    time_s: np.ndarray,
    depths_cm: np.ndarray,
    slope_deg: float,
    rain_mm_10min: np.ndarray,
    lulc: dict,
    soil_onehot: np.ndarray,
    met: dict,
    aspect_deg: np.ndarray,
    theta0_by_depth: np.ndarray,            # ✅ NEW (nZ,)
    soil_thickness_m: float = 3.0,
    thickness_train_norm: float = THICKNESS_TRAIN_NORM,
) -> np.ndarray:
    """
    Build feature matrix for point prediction, 20 columns (must match training):
    [t, x, z, slope, rain, runoff, FVC, ISA, soil5, met4, soil_thickness, aspect, theta0]
    """

    T_hr = (time_s.astype(np.float32) / 3600.0)

    Zm = depths_cm.astype(np.float32) / 100.0
    z_norm = Zm / max(float(thickness_train_norm), 1e-6)

    FVC = lulc['FVC'].astype(np.float32)
    ISA = lulc['ISA'].astype(np.float32)
    r10 = rain_mm_10min.astype(np.float32)

    tc  = met['temp_C'].astype(np.float32)
    rh  = met['RH_pct'].astype(np.float32)
    ws  = met['wind_ms'].astype(np.float32)
    wd  = met['wind_dir_deg'].astype(np.float32)

    asp = aspect_deg.astype(np.float32)

    nT = len(T_hr)
    nZ = len(z_norm)

    TT, ZZ = np.meshgrid(T_hr, z_norm)     # (nZ, nT)
    FFC, IISA = np.meshgrid(FVC, z_norm)
    RR  = np.meshgrid(r10, z_norm)[0]
    TC  = np.meshgrid(tc, z_norm)[0]
    RH  = np.meshgrid(rh, z_norm)[0]
    WS  = np.meshgrid(ws, z_norm)[0]
    WD  = np.meshgrid(wd, z_norm)[0]
    AD  = np.meshgrid(asp, z_norm)[0]

    x_col   = np.zeros_like(TT, dtype=np.float32)
    slope_c = np.full_like(TT, float(slope_deg), dtype=np.float32)
    thick   = np.full_like(TT, float(soil_thickness_m), dtype=np.float32)

    w = soil_onehot.astype(np.float32)
    wcp = np.full_like(TT, w[0], dtype=np.float32)
    wgr = np.full_like(TT, w[1], dtype=np.float32)
    wsd = np.full_like(TT, w[2], dtype=np.float32)
    wdc = np.full_like(TT, w[3], dtype=np.float32)
    wsc = np.full_like(TT, w[4], dtype=np.float32)

    # runoff (mm/10min)
    soil_w_T = np.tile(soil_onehot.reshape(1, 5), (nT, 1)).astype(np.float32)
    runoff_10min = RunoffCalculator.runoff_generation_mm10(
        rain_mm10=r10,
        soil_w=soil_w_T,
        FVC=FVC,
        ISA=ISA,
        slope_deg=np.full_like(r10, slope_deg),
        cap_scale=1.0
    ).astype(np.float32)
    runoff = np.repeat(runoff_10min[np.newaxis, :], nZ, axis=0)

    # ✅ theta0 broadcast: (nZ,) -> (nZ,nT)
    theta0_by_depth = np.asarray(theta0_by_depth, dtype=np.float32).reshape(nZ, 1)
    TH0 = np.repeat(theta0_by_depth, nT, axis=1)

    feat = np.stack([
        TT, x_col, ZZ,
        slope_c, RR, runoff,
        FFC, IISA,
        wcp, wgr, wsd, wdc, wsc,
        TC, RH, WS, WD,
        thick,
        AD,
        TH0                         # ✅ NEW col 19
    ], axis=2).reshape(nZ * nT, 20).astype(np.float32)

    return feat

def predict_from_excel_and_compare(model,
                                   excel_path: str,
                                   compare_time_s: float = None,
                                   out_dir: str = "predictions",
                                   soil_thickness_m: float = 3.0,
                                   save_heatmap: bool = True,
                                   save_profile_plot: bool = True,
                                   save_grids_csv: bool = True,
                                   save_profile_csv: bool = True):
    """
    Point prediction + comparison for a single Excel file.

    New features:
      - Prediction/heatmaps are on a dense depth grid: 0–50 cm at 1 cm step.
      - Error heatmap (Predicted - Measured) on the same grid.
      - Output filenames include the Excel stem to avoid overwriting.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Use stem so 10mmh and 90mmh do not overwrite each other
    stem = os.path.splitext(os.path.basename(excel_path))[0]

    # --- parse soil & slope from filename ---
    soil, slope = _parse_soil_slope_from_filename(excel_path)
    w_soil = _soil_onehot(soil)

    # --- read forcing + observations from Excel ---
    series = _read_excel_rowwise_robust(excel_path)
    time_s   = series['time_s'].astype(np.float32)          # (T,)
    rain_10  = series['rain_mm_10min'].astype(np.float32)   # (T,)
    met      = series['met']
    lulc     = series['lulc']
    aspect_deg = series.get('aspect_deg', np.zeros_like(time_s))

    # Observations (may be at only a few depths)
    obs_depths_cm = series['depths_cm_obs'].astype(np.float32)     # (nZ_obs,)
    obs_grid      = series['water_content_obs'].astype(np.float32) # (nZ_obs, T)

    # ------------------------------------------------------------------
    # 1) Choose depth grid for prediction/heatmaps
    #    Make it reach the maximum observed depth in THIS file
    # ------------------------------------------------------------------
    have_obs = (obs_depths_cm.size > 0) and (obs_grid.size > 0)

    if have_obs:
        max_obs_cm = float(np.nanmax(obs_depths_cm))
    else:
        max_obs_cm = 50.0  # fallback

    fine_max = 50.0
    fine_step = 1.0
    coarse_step = 5.0

    if max_obs_cm <= fine_max:
        depths_cm = np.arange(0.0, max_obs_cm + fine_step, fine_step, dtype=np.float32)
    else:
        depths_fine = np.arange(0.0, fine_max + fine_step, fine_step, dtype=np.float32)
        depths_coarse = np.arange(fine_max + coarse_step, max_obs_cm + coarse_step, coarse_step, dtype=np.float32)
        depths_cm = np.unique(np.concatenate([depths_fine, depths_coarse])).astype(np.float32)

    have_obs = (obs_depths_cm.size > 0) and (obs_grid.size > 0)
    if have_obs:
        theta0_at_obs = obs_grid[:, 0]  # first time column as initial
        theta0_preddepth = np.interp(
            depths_cm, obs_depths_cm, theta0_at_obs,
            left=theta0_at_obs[0], right=theta0_at_obs[-1]
        ).astype(np.float32)
    else:
        theta0_preddepth = np.full_like(
            depths_cm, SoilParameters.TABLE[soil]['theta_r'], dtype=np.float32
        )


    # Build feature matrix (N = nZ * T, 18 features)
    X = _build_point_features_20(time_s, depths_cm, slope,
                                 rain_10, lulc, w_soil, met,
                                 aspect_deg=aspect_deg,
                                 soil_thickness_m=soil_thickness_m,
                                 theta0_by_depth=theta0_preddepth,)

    # ------------------------------------------------------------------
    # 2) Run model and reshape to [nZ_depths, nT_times]
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        Y = model(torch.from_numpy(X).to(device)).cpu().numpy().astype(np.float32)

    pred_grid = Y.reshape(len(depths_cm), len(time_s))  # [nZ, T]

    # Save predicted grid on 0–50 cm every 1 cm
    if save_grids_csv:
        df_pred = pd.DataFrame(pred_grid, index=depths_cm, columns=time_s)
        df_pred.index.name = "Depth_cm"
        df_pred.columns.name = "Time_s"
        df_pred.to_csv(os.path.join(out_dir, f"pred_grid_{stem}.csv"))

    # ------------------------------------------------------------------
    # 3) If measured data exist, build interpolated obs grid on same depths
    #    and compute an error grid: Predicted - Measured
    # ------------------------------------------------------------------
    have_obs = (obs_depths_cm.size > 0) and (obs_grid.size > 0)

    if have_obs:
        nZ_heat = len(depths_cm)
        nT = len(time_s)
        obs_interp = np.zeros((nZ_heat, nT), dtype=np.float32)

        if len(obs_depths_cm) == 1:
            # Only one observed depth: fill each column with that value
            for ti in range(nT):
                obs_interp[:, ti] = obs_grid[0, ti]
        else:
            # Interpolate measured profile to every 1 cm for each time
            for ti in range(nT):
                obs_prof = obs_grid[:, ti]
                # Linear interpolation; clamp outside range with end values
                obs_interp[:, ti] = np.interp(
                    depths_cm,
                    obs_depths_cm,
                    obs_prof,
                    left=obs_prof[0],
                    right=obs_prof[-1]
                ).astype(np.float32)

        # Error grid on 0–50 cm, every time step
        err_grid = pred_grid - obs_interp

        if save_grids_csv:
            df_err = pd.DataFrame(err_grid, index=depths_cm, columns=time_s)
            df_err.index.name = "Depth_cm"
            df_err.columns.name = "Time_s"
            df_err.to_csv(os.path.join(out_dir, f"error_grid_{stem}.csv"))
    else:
        err_grid = None

    # ------------------------------------------------------------------
    # 4) Heatmaps: predicted VWC, and error if obs exist
    # ------------------------------------------------------------------
    if save_heatmap:
        # Predicted VWC heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        pcm = ax.pcolormesh(time_s, depths_cm, pred_grid,
                            cmap='viridis', shading='auto')
        ax.invert_yaxis()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Depth (cm)')
        ax.set_title(f'{soil} soil | slope {slope:g}° | Predicted VWC')
        fig.colorbar(pcm, ax=ax, label='VWC (–)')
        fig.savefig(os.path.join(out_dir,
                                 f"heatmap_pred_{stem}.png"),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Error heatmap (Predicted - Measured) if we have measurements
        if have_obs:
            fig, ax = plt.subplots(figsize=(10, 8))
            pcm = ax.pcolormesh(time_s, depths_cm, err_grid,
                                cmap='RdBu_r', shading='auto')
            ax.invert_yaxis()
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Depth (cm)')
            ax.set_title(f'{soil} soil | slope {slope:g}° | Error (Predicted - Measured)')
            fig.colorbar(pcm, ax=ax, label='Error in VWC (–)')
            fig.savefig(os.path.join(out_dir,
                                     f"heatmap_error_{stem}.png"),
                        dpi=200, bbox_inches='tight')
            plt.close(fig)

    # ------------------------------------------------------------------
    # 5) Profile comparison at selected time (still at MEASURED depths)
    # ------------------------------------------------------------------
    if not have_obs:
        print(f"[info] No monitoring matrix in {os.path.basename(excel_path)}")
        return {'soil': soil, 'slope': slope}

    # Choose comparison time
    if compare_time_s is None:
        compare_time_s = float(np.median(time_s))
    ti = int(np.argmin(np.abs(time_s - compare_time_s)))
    t_sel = float(time_s[ti])

    # Predicted profile at observed depths (not 0–50 grid)
    if np.array_equal(depths_cm, obs_depths_cm):
        pred_prof = pred_grid[:, ti]
    else:
        pred_prof = np.interp(obs_depths_cm,
                              depths_cm,
                              pred_grid[:, ti]).astype(np.float32)

    obs_prof = obs_grid[:, ti]
    diff = pred_prof - obs_prof

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    ss_res = float(np.sum((obs_prof - pred_prof) ** 2))
    ss_tot = float(np.sum((obs_prof - np.mean(obs_prof)) ** 2))
    if ss_tot < 1e-12:
        r2 = np.nan   # not meaningful if measured profile is almost constant
    else:
        r2 = 1.0 - ss_res / ss_tot

    # Profile plot
    if save_profile_plot:
        plt.figure(figsize=(6, 7))
        plt.plot(obs_prof, obs_depths_cm, label='Measured', lw=2)
        plt.plot(pred_prof, obs_depths_cm, '--', label='Predicted', lw=2)
        plt.gca().invert_yaxis()
        plt.xlabel('VWC (–)')
        plt.ylabel('Depth (cm)')
        plt.title(f'{soil} soil | slope {slope:g}° | t={int(t_sel)} s')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(out_dir,
                                 f"profile_{stem}_t{int(t_sel)}s.png"),
                    dpi=200, bbox_inches='tight')
        plt.close()

    # Profile CSV (at comparison time)
    if save_profile_csv:
        df_cmp = pd.DataFrame({
            'Depth_cm': obs_depths_cm,
            'Measured_VWC': obs_prof,
            'Predicted_VWC': pred_prof,
            'Abs_Error': np.abs(diff)
        })
        df_cmp.to_csv(os.path.join(out_dir,
                                   f"profile_{stem}_t{int(t_sel)}s.csv"),
                      index=False)

    print(f"[compare] {soil} soil | slope={slope:g}° | t={int(t_sel)}s "
          f"| MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

    return {
        'soil': soil,
        'slope': slope,
        't_sel': t_sel,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def batch_compare_folder(model,
                         data_dir: str,
                         compare_time_s: float = None,
                         out_dir: str = "predictions"):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for fn in sorted(os.listdir(data_dir)):
        if fn.lower().endswith(".xlsx"):
            path = os.path.join(data_dir, fn)
            try:
                res = predict_from_excel_and_compare(
                    model, path, compare_time_s=compare_time_s, out_dir=out_dir
                )
                rows.append({
                    'file': fn,
                    'soil': res.get('soil'),
                    'slope': res.get('slope'),
                    't_sel_s': res.get('t_sel', np.nan),
                    'MAE': res.get('mae', np.nan),
                    'RMSE': res.get('rmse', np.nan),
                    'R2': res.get('r2', np.nan)
                })
            except Exception as e:
                print(f"[error] {fn}: {e}")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "point_compare_summary.csv"), index=False)
        print(f"Saved summary → {os.path.join(out_dir, 'point_compare_summary.csv')}")
    return rows

# ==========================================================
# Main
# ==========================================================
def main():
    project_dir = "/content/drive/MyDrive/PINN"    # <<< UPDATED path

    # Optional: generate a fresh Excel template and exit
    GENERATE_TEMPLATE_ONLY = False
    if GENERATE_TEMPLATE_ONLY:
        Visualizer.export_excel_template(project_dir,
                                        filename="PINN_training_template.xlsx",
                                        T=13,
                                        depths_cm=(10, 30, 60, 100))
        return

    # --- load all required inputs (rasters optional for training, required for map prediction) ---
    dl = DataLoader(project_dir)
    ok = dl.load_all()
    if not ok:
        print("Failed to load some inputs. Training can still proceed if Excel loaded.")
    if not dl.excel_data:
        raise RuntimeError("No Excel training files found. Put .xlsx under project_dir or project_dir/train/")

    # --- build & train ---
    # TUNED: Use new architecture with depth embedding
    model = EnhancedPINN(
        hidden=[128, 256, 256, 128],
        activation="silu",      # Better gradients than tanh
        out_temp=8.0,           # Less saturation
        n_depth_freqs=6,        # Fourier depth embedding
    )
    trainer = PINNTrainer(model, dl)
    # TUNED: More epochs for better data fitting before physics
    loss_hist = trainer.train(epochs=3000)

    # --- plot loss to outputs/ ---
    vis = Visualizer(project_dir)
    vis.plot_training_loss(loss_hist)

    # --- after training: compare on each Excel file using its own rainfall/time/soil/slope/aspect ---
    excel_folder = os.path.join(project_dir, "train") if os.path.isdir(os.path.join(project_dir, "train")) else project_dir
    out_dir = os.path.join(project_dir, "outputs_point")
    os.makedirs(out_dir, exist_ok=True)

    # choose comparison time (None → median time in each file)
    batch_compare_folder(model, excel_folder, compare_time_s=None, out_dir=out_dir)

    print("✅ Training finished and per-file comparisons saved to:", out_dir)

# === keep the guard AT THE VERY END of the file ===
if __name__ == "__main__":
    main()