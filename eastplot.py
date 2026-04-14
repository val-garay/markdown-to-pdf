# ============================================================
# SINGLE CLEAN SCRIPT (Colab) — Prediction-Only OR Validation
# - If MONITOR_FILE is empty -> prediction-only
# - If MONITOR_FILE is provided -> prediction + compare (NO obs interpolation)
# - Dense depth grid (more near surface, less at depth)
# - ALL PLOTS:
#     (Validation mode) 1) profiles (pred line + obs points) at selected times
#                       2) time series at each monitored depth
#                       3) scatter obs vs pred
#                       4) heatmap (depth vs time)
#     (Prediction-only) 1) profiles at selected times
#                       2) heatmap
# - Depth axis: 0 m at top (depth increases downward)
# ============================================================

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---------------- USER SETTINGS ----------------
PARAM_FILE   = "/content/drive/MyDrive/PINN/GR_s0_L1_3.xlsx"   # forcing/parameters
MONITOR_FILE = "/content/drive/MyDrive/PINN/GR_s0_L1_3.xlsx"  # "" for prediction-only, or set path for validation

CKPT_PATH    = "/content/drive/MyDrive/PINN/best_pinn_model.pth"
OUT_DIR      = "/content/drive/MyDrive/point_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

THICKNESS_TRAIN_NORM = 3.0  # must match training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ---- Prediction-only controls ----
PRED_MAX_DEPTH_M   = 3.0
PRED_THETA0_CONST  = 0.20
CUSTOM_TIMES_HR    = [1, 12, 24, 6, 48]   # edit freely


# ---------------- Soil Parameters (must match training) ----------------
class SoilParameters:
    CP = {'theta_s':0.56, 'theta_r':0.016, 'alpha':0.018,  'n':1.17,  'm':0.145, 'Ksatx':4.94e-8, 'Ksatz':4.94e-8}
    SC = {'theta_s':0.37, 'theta_r':0.01,  'alpha':0.0290, 'n':1.492, 'm':0.329, 'Ksatx':0.000142,'Ksatz':0.000142}
    SD = {'theta_s':0.43, 'theta_r':0.013, 'alpha':0.811,  'n':1.48,  'm':0.324, 'Ksatx':1.73e-5,'Ksatz':1.73e-5}
    DC = {'theta_s':0.42, 'theta_r':0.016, 'alpha':0.787,  'n':1.55,  'm':0.355, 'Ksatx':2.05e-5,'Ksatz':2.05e-5}
    GR = {'theta_s':0.48, 'theta_r':0.024, 'alpha':0.0035, 'n':1.36,  'm':0.265, 'Ksatx':9.78e-7,'Ksatz':9.78e-7}
    TABLE = {'CP': CP, 'GR': GR, 'SD': SD, 'DC': DC, 'SC': SC}
    ORDER = ['CP','GR','SD','DC','SC']

def _soil_onehot(tag: str) -> np.ndarray:
    tag = tag.upper().strip()
    w = np.zeros(5, dtype=np.float32)
    if tag not in SoilParameters.ORDER:
        tag = "CP"
    w[SoilParameters.ORDER.index(tag)] = 1.0
    return w

def parse_slope_from_filename(path: str, default=0.0) -> float:
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'[sS](\d+)', base)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
    return float(default)

def parse_soil_from_filename(path: str, default="CP") -> str:
    stem = os.path.splitext(os.path.basename(path))[0].upper()
    tokens = re.split(r'[^A-Z0-9]+', stem)
    tokens = [t for t in tokens if t]
    for tag in SoilParameters.ORDER:
        if tag in tokens:
            return tag
    return default


# ---------------- Runoff (same logic as your model) ----------------
class RunoffCalculator:
    @staticmethod
    def _infil_capacity_mm10(soil_w, FVC, ISA, slope_deg, cap_scale=1.0):
        Ksatz_vec = np.array([
            SoilParameters.CP['Ksatz'],
            SoilParameters.GR['Ksatz'],
            SoilParameters.SD['Ksatz'],
            SoilParameters.DC['Ksatz'],
            SoilParameters.SC['Ksatz'],
        ], dtype=np.float32)

        Ksatz = (soil_w * Ksatz_vec[None, :]).sum(axis=-1).astype(np.float32)
        cap0 = Ksatz * 600.0 * 1000.0  # mm/10min

        f_isa = np.clip(1.0 - ISA, 0.0, 1.0)
        f_fvc = 0.3 + 0.7 * np.clip(FVC, 0.0, 1.0)

        srad = np.deg2rad(np.clip(slope_deg, 0.0, 89.0))
        f_slope = 1.0 / (1.0 + np.tan(srad))

        cap = cap_scale * cap0 * f_isa * f_fvc * f_slope
        return np.clip(cap, 0.0, None).astype(np.float32)

    @classmethod
    def runoff_generation_mm10(cls, rain_mm10, soil_w, FVC, ISA, slope_deg, cap_scale=1.0):
        cap = cls._infil_capacity_mm10(soil_w, FVC, ISA, slope_deg, cap_scale=cap_scale)
        return np.maximum(rain_mm10 - cap, 0.0).astype(np.float32)


# ---------------- Model (MUST match checkpoint keys) ----------------
class EnhancedPINN(nn.Module):
    def __init__(self, hidden=[128,256,256,128], activation="silu", out_temp=8.0, n_depth_freqs=6):
        super().__init__()
        self.input_dim = 20
        self.out_temp = float(out_temp)
        self.n_depth_freqs = int(n_depth_freqs)

        # keys that existed in your checkpoint:
        self.log_Ksatx_scale = nn.Parameter(torch.tensor(0.0))
        self.log_Ksatz_scale = nn.Parameter(torch.tensor(0.0))
        self.lambda_input = nn.Parameter(torch.ones(self.input_dim, 1))

        self.depth_embed_dim = 1 + 2*self.n_depth_freqs
        self.interaction_dim = 7
        self.expanded_input_dim = (self.input_dim - 1) + self.depth_embed_dim + self.interaction_dim

        Act = nn.SiLU if activation.lower() in ["silu","swish"] else nn.Tanh
        self.fc1 = nn.Linear(self.expanded_input_dim, hidden[0]); self.act1 = Act()
        self.fc2 = nn.Linear(hidden[0], hidden[1]); self.act2 = Act()
        self.fc3 = nn.Linear(hidden[1] + self.depth_embed_dim + self.interaction_dim, hidden[2]); self.act3 = Act()
        self.fc4 = nn.Linear(hidden[2], hidden[3]); self.act4 = Act()
        self.fc_out = nn.Linear(hidden[3], 1)

    def _depth_embedding(self, z):
        emb = [z]
        for i in range(self.n_depth_freqs):
            freq = (2.0**i) * torch.pi
            emb += [torch.sin(freq*z), torch.cos(freq*z)]
        return torch.cat(emb, dim=-1)

    def forward(self, x):
        # (lambda/log scales exist for checkpoint compatibility; not applied here)
        t = x[:,0:1]
        z = x[:,2:3]
        slope = x[:,3:4]
        rain = x[:,4:5]

        z_embed = self._depth_embedding(z)

        slope_rad = slope * torch.pi / 180.0
        cos_slope = torch.cos(slope_rad)
        sin_slope = torch.sin(slope_rad)
        t_norm = t / 20.0

        t_z = t*z
        slope_z = slope/90.0*z
        rain_z = rain/10.0*z
        cos_slope_z = cos_slope*z
        sin_slope_t_z = sin_slope*t_norm*z
        early_rain_z = (1.0 - t_norm)*rain/10.0*z
        rain_time_ratio = (rain/10.0)/(t_norm+0.1)
        rain_time_slope_z = rain_time_ratio*sin_slope*z

        inter = torch.cat([t_z,slope_z,rain_z,cos_slope_z,sin_slope_t_z,early_rain_z,rain_time_slope_z], dim=-1)

        x_no_z = torch.cat([x[:,:2], x[:,3:]], dim=-1)
        x_exp = torch.cat([x_no_z[:,:2], z_embed, inter, x_no_z[:,2:]], dim=-1)

        h1 = self.act1(self.fc1(x_exp))
        h2 = self.act2(self.fc2(h1))
        h2s = torch.cat([h2, z_embed, inter], dim=-1)
        h3 = self.act3(self.fc3(h2s))
        h4 = self.act4(self.fc4(h3))
        raw = self.fc_out(h4)

        w = x[:,8:13]  # soil weights
        theta_r_vec = torch.tensor([
            SoilParameters.CP['theta_r'],SoilParameters.GR['theta_r'],SoilParameters.SD['theta_r'],
            SoilParameters.DC['theta_r'],SoilParameters.SC['theta_r']
        ], device=x.device).view(1,5)
        theta_s_vec = torch.tensor([
            SoilParameters.CP['theta_s'],SoilParameters.GR['theta_s'],SoilParameters.SD['theta_s'],
            SoilParameters.DC['theta_s'],SoilParameters.SC['theta_s']
        ], device=x.device).view(1,5)

        theta_r = (w*theta_r_vec).sum(dim=1, keepdim=True)
        theta_s = (w*theta_s_vec).sum(dim=1, keepdim=True)

        y = torch.sigmoid(raw / self.out_temp)
        theta = theta_r + (theta_s-theta_r)*y
        eps = 1e-6
        return torch.clamp(theta, theta_r+eps, theta_s-eps)


# ---------------- Excel readers ----------------
def read_forcing_excel(excel_path: str) -> dict:
    df = pd.read_excel(excel_path, header=None)

    def row_vals(r):
        return pd.to_numeric(r[1:], errors="coerce").to_numpy(dtype=np.float64)

    time_vals = None
    rain_10 = None
    rain_h = None
    met = {'temp_C':None,'RH_pct':None,'wind_ms':None,'wind_dir_deg':None}
    lulc = {'FVC':None,'ISA':None}
    aspect = None
    slope_series = None

    for r in range(len(df)):
        lab = df.iloc[r,0]
        if not isinstance(lab, str):
            continue
        k = lab.strip().lower()

        if "time" in k:
            time_vals = row_vals(df.iloc[r,:]); continue
        if k == "rain_mm_10min":
            rain_10 = row_vals(df.iloc[r,:]); continue
        if k == "rain_mm_h":
            rain_h = row_vals(df.iloc[r,:]); continue

        if k == "temp_c": met["temp_C"] = row_vals(df.iloc[r,:]); continue
        if k == "rh_pct": met["RH_pct"] = row_vals(df.iloc[r,:]); continue
        if k == "wind_ms": met["wind_ms"] = row_vals(df.iloc[r,:]); continue
        if k == "wind_dir_deg": met["wind_dir_deg"] = row_vals(df.iloc[r,:]); continue
        if k == "fvc": lulc["FVC"] = row_vals(df.iloc[r,:]); continue
        if k == "isa": lulc["ISA"] = row_vals(df.iloc[r,:]); continue
        if k in ("aspect","aspect_deg"): aspect = row_vals(df.iloc[r,:]); continue
        if k in ("slope_deg","slope"): slope_series = row_vals(df.iloc[r,:]); continue

    if time_vals is None:
        raise ValueError("PARAM file missing time row (time/time_sec).")

    t_s = np.asarray(time_vals, dtype=np.float32)
    T = len(t_s)

    # dt for conversion if rain is mm/h
    if T >= 2:
        dt0 = float(np.nanmedian(np.diff(t_s)))
        if not np.isfinite(dt0) or dt0 <= 0:
            dt0 = 600.0
    else:
        dt0 = 600.0

    if rain_10 is None and rain_h is None:
        raise ValueError("PARAM file missing rainfall row (rain_mm_10min or rain_mm_h).")

    if rain_10 is None:
        rain_h = np.asarray(rain_h, dtype=np.float32)
        if len(rain_h) == T:
            mm_step = rain_h * (dt0/3600.0)
            rain_10 = mm_step * (600.0/dt0)  # mm/10min-equiv
        else:
            rain_10 = rain_h/6.0
    else:
        rain_10 = np.asarray(rain_10, dtype=np.float32)

    def fill(arr, default):
        if arr is None:
            return np.full(T, default, dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr) != T:
            raise ValueError("PARAM series length mismatch with time.")
        return arr

    return {
        "time_s": t_s,
        "rain_mm10": rain_10,
        "temp_C": fill(met["temp_C"], 25.0),
        "RH_pct": fill(met["RH_pct"], 70.0),
        "wind_ms": fill(met["wind_ms"], 1.0),
        "wind_dir_deg": fill(met["wind_dir_deg"], 0.0),
        "FVC": np.clip(fill(lulc["FVC"], 0.5), 0.0, 1.0),
        "ISA": np.clip(fill(lulc["ISA"], 0.2), 0.0, 1.0),
        "aspect_deg": fill(aspect, 0.0),
        "slope_series": fill(slope_series, np.nan) if slope_series is not None else None,
        "dt0": float(dt0),
    }

def read_monitoring_excel(excel_path: str) -> dict:
    df = pd.read_excel(excel_path, header=None)

    def row_vals(r):
        return pd.to_numeric(r[1:], errors="coerce").to_numpy(dtype=np.float64)

    time_vals = None
    depths_cm = []
    moist_rows = []

    for r in range(len(df)):
        lab = df.iloc[r,0]
        if lab is None or (isinstance(lab,float) and np.isnan(lab)):
            continue

        if isinstance(lab,str) and ("time" in lab.lower()):
            time_vals = row_vals(df.iloc[r,:]).astype(np.float32)
            continue

        try:
            dcm = float(str(lab).strip())
            vals = row_vals(df.iloc[r,:]).astype(np.float32)
            if not np.isnan(vals).all():
                depths_cm.append(dcm)
                moist_rows.append(vals)
        except:
            pass

    if time_vals is None:
        raise ValueError("MONITOR file missing time row (time/time_sec).")
    if len(depths_cm) == 0:
        raise ValueError("MONITOR file has no numeric depth rows in column A.")

    t_s = np.asarray(time_vals, dtype=np.float32)
    obs = np.vstack(moist_rows).astype(np.float32)

    depths_cm = np.asarray(depths_cm, dtype=np.float32)
    order = np.argsort(depths_cm)
    depths_cm = depths_cm[order]
    obs = obs[order, :]

    T = len(t_s)
    obs = obs[:, :T]
    return {"time_s": t_s, "depths_cm": depths_cm, "theta_obs": obs}
def rain_mm10_to_mmh(rain_mm10: np.ndarray) -> np.ndarray:
    """Convert mm/10min-equivalent to mm/h for plotting."""
    rain_mm10 = np.asarray(rain_mm10, dtype=float)
    return rain_mm10 * 6.0

# ---------------- Depth grid (dense near surface) ----------------
def make_depth_grid_m(max_depth_m: float) -> np.ndarray:
    max_depth_m = float(max_depth_m)
    a = np.arange(0.0, min(0.30, max_depth_m)+1e-9, 0.01, dtype=np.float32)
    b = np.arange(0.35, min(1.00, max_depth_m)+1e-9, 0.05, dtype=np.float32) if max_depth_m > 0.30 else np.array([], dtype=np.float32)
    c = np.arange(1.10, max_depth_m+1e-9, 0.10, dtype=np.float32) if max_depth_m > 1.00 else np.array([], dtype=np.float32)
    z = np.unique(np.concatenate([a, b, c]))
    return z

def make_depth_grid_custom(max_depth_m: float) -> np.ndarray:
    z = make_depth_grid_m(max_depth_m)
    if z.size == 0 or float(z[0]) != 0.0:
        z = np.unique(np.concatenate([np.array([0.0], dtype=np.float32), z]))
    return z

def theta0_profile_constant(depths_m: np.ndarray, theta0=0.20) -> np.ndarray:
    return np.full_like(np.asarray(depths_m, dtype=np.float32), float(theta0), dtype=np.float32)

def rain_mm10_to_mmh(rain_mm10: np.ndarray) -> np.ndarray:
    """Convert mm/10min-equivalent to mm/h for plotting."""
    rain_mm10 = np.asarray(rain_mm10, dtype=float)
    return rain_mm10 * 6.0

# ---------------- Build feature matrix (20 cols) ----------------
def build_features_20(time_s, depths_m, soil_tag, slope_deg, forcing, theta0_by_depth, soil_thickness_m=THICKNESS_TRAIN_NORM):
    time_s = np.asarray(time_s, dtype=np.float32)
    depths_m = np.asarray(depths_m, dtype=np.float32)
    nT = len(time_s)
    nZ = len(depths_m)

    t_hr = time_s / 3600.0
    z_norm = depths_m / float(THICKNESS_TRAIN_NORM)

    TT, ZZ = np.meshgrid(t_hr, z_norm)  # (nZ,nT)

    def mesh_time(v1d):
        return np.meshgrid(v1d.astype(np.float32), z_norm)[0]

    R10 = mesh_time(forcing["rain_mm10"])
    FVC = mesh_time(forcing["FVC"])
    ISA = mesh_time(forcing["ISA"])
    TC  = mesh_time(forcing["temp_C"])
    RH  = mesh_time(forcing["RH_pct"])
    WS  = mesh_time(forcing["wind_ms"])
    WD  = mesh_time(forcing["wind_dir_deg"])
    ASP = mesh_time(forcing["aspect_deg"])

    w = _soil_onehot(soil_tag)
    wcp = np.full_like(TT, w[0], dtype=np.float32)
    wgr = np.full_like(TT, w[1], dtype=np.float32)
    wsd = np.full_like(TT, w[2], dtype=np.float32)
    wdc = np.full_like(TT, w[3], dtype=np.float32)
    wsc = np.full_like(TT, w[4], dtype=np.float32)

    soil_w_T = np.tile(w.reshape(1,5), (nT,1)).astype(np.float32)
    runoff_1d = RunoffCalculator.runoff_generation_mm10(
        rain_mm10=forcing["rain_mm10"].astype(np.float32),
        soil_w=soil_w_T,
        FVC=forcing["FVC"].astype(np.float32),
        ISA=forcing["ISA"].astype(np.float32),
        slope_deg=np.full(nT, float(slope_deg), dtype=np.float32),
        cap_scale=1.0
    )
    RUN = np.repeat(runoff_1d.reshape(1,nT), nZ, axis=0)

    th0 = np.asarray(theta0_by_depth, dtype=np.float32).reshape(nZ,1)
    TH0 = np.repeat(th0, nT, axis=1)

    X = np.stack([
        TT,                                  # 0 t_hr
        np.zeros_like(TT, dtype=np.float32), # 1 x
        ZZ,                                  # 2 z_norm
        np.full_like(TT, float(slope_deg), dtype=np.float32), # 3 slope
        R10,                                 # 4 rain
        RUN,                                 # 5 runoff
        FVC,                                 # 6 FVC
        ISA,                                 # 7 ISA
        wcp,wgr,wsd,wdc,wsc,                 # 8..12 soil
        TC,RH,WS,WD,                         # 13..16 met
        np.full_like(TT, float(soil_thickness_m), dtype=np.float32), # 17 thickness
        ASP,                                 # 18 aspect
        TH0                                  # 19 theta0
    ], axis=2).reshape(nZ*nT, 20).astype(np.float32)

    return X


# ---------------- Utils: plotting helpers ----------------
def closest_time_indices(time_s: np.ndarray, custom_times_hr):
    time_s = np.asarray(time_s, dtype=float)
    idxs = []
    for t_hr in custom_times_hr:
        t_target = float(t_hr) * 3600.0
        idxs.append(int(np.argmin(np.abs(time_s - t_target))))
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            out.append(i); seen.add(i)
    return out

def plot_heatmap_predicted(time_s, depths_pred_m, pred_grid, title="Predicted θ heatmap"):
    time_s = np.asarray(time_s, dtype=float)
    depths_pred_m = np.asarray(depths_pred_m, dtype=float)
    pred_grid = np.asarray(pred_grid, dtype=float)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        pred_grid,
        aspect="auto",
        origin="upper",
        extent=[
            time_s[0] / 3600.0,
            time_s[-1] / 3600.0,
            float(depths_pred_m[-1]),
            float(depths_pred_m[0]),
        ]
    )
    plt.colorbar(label="Predicted θ")
    plt.xlabel("Time (hours)")
    plt.ylabel("Depth (m)")
    plt.title(title)
    plt.show()

def plot_profiles_multiple_times(time_s, depths_pred_m, pred_grid, custom_times_hr, title="Profiles at selected times (0 m at top)"):
    idxs = closest_time_indices(time_s, custom_times_hr)
    plt.figure(figsize=(7, 9))
    for ti in idxs:
        t_hr_actual = float(time_s[ti]) / 3600.0
        plt.plot(pred_grid[:, ti], depths_pred_m, linewidth=2, label=f"Pred t={t_hr_actual:.2f} h")
    plt.ylim(0, float(np.nanmax(depths_pred_m)))
    plt.gca().invert_yaxis()
    plt.xlabel("VWC (θ)")
    plt.ylabel("Depth (m)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.show()

def plot_profiles_with_obs_multiple_times(time_s, depths_pred_m, pred_grid, depths_obs_m, theta_obs, custom_times_hr, title="Profile comparison (0 m at top)"):
    idxs = closest_time_indices(time_s, custom_times_hr)
    plt.figure(figsize=(7, 9))
    for ti in idxs:
        t_hr_actual = float(time_s[ti]) / 3600.0
        plt.plot(pred_grid[:, ti], depths_pred_m, linewidth=2, label=f"Pred t={t_hr_actual:.2f} h")
        plt.plot(theta_obs[:, ti], depths_obs_m, "o", markersize=5, label=f"Obs t={t_hr_actual:.2f} h")
    plt.ylim(0, float(np.nanmax(depths_pred_m)))
    plt.gca().invert_yaxis()
    plt.xlabel("VWC (θ)")
    plt.ylabel("Depth (m)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(ncols=2, fontsize=9)
    plt.show()

def plot_timeseries_each_depth_with_rain(
    time_s,
    rain_input,
    depths_obs_cm,
    theta_obs,
    pred_at_obs,
    title_prefix="Time series at depth",
    rain_label="Rain (input unit)",
    vwc_min=0.020,          # NEW
    vwc_max=0.600           # NEW
):
    import numpy as np
    import matplotlib.pyplot as plt

    time_s = np.asarray(time_s, dtype=float)
    t_hr = time_s / 3600.0

    rain_input = np.asarray(rain_input, dtype=float)
    depths_obs_cm = np.asarray(depths_obs_cm, dtype=float)
    theta_obs = np.asarray(theta_obs, dtype=float)
    pred_at_obs = np.asarray(pred_at_obs, dtype=float)

    if len(rain_input) != len(t_hr):
        raise ValueError("Rain length must equal time length.")

    for i in range(len(depths_obs_cm)):

        fig, ax1 = plt.subplots(figsize=(9, 5))

        # ---- θ axis (left) ----
        line_obs, = ax1.plot(
            t_hr, theta_obs[i, :],
            "o-", linewidth=1.5, markersize=4,
            label="Observed θ"
        )
        line_pred, = ax1.plot(
            t_hr, pred_at_obs[i, :],
            "-", linewidth=2,
            label="Predicted θ"
        )

        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("VWC (θ)")
        ax1.set_ylim(vwc_min, vwc_max)   # ✅ FIXED RANGE
        ax1.grid(alpha=0.3)

        # ---- Rain axis (right) ----
        ax2 = ax1.twinx()
        line_rain, = ax2.plot(
            t_hr, rain_input,
            linewidth=2,
            linestyle="--",
            color="green",
            label=rain_label
        )
        ax2.set_ylabel(rain_label)

        # ---- Combined legend ----
        lines = [line_obs, line_pred, line_rain]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")

        plt.title(f"{title_prefix} = {depths_obs_cm[i]:g} cm")
        plt.tight_layout()
        plt.show()

def plot_scatter_obs_vs_pred(df_cmp, R2_all=None, title="Observed vs Predicted"):
    x = df_cmp["theta_obs"].to_numpy(dtype=float)
    y = df_cmp["theta_pred"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size == 0:
        print("No finite obs/pred pairs for scatter.")
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=12)
    mn = float(min(x.min(), y.min()))
    mx = float(max(x.max(), y.max()))
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("Observed θ")
    plt.ylabel("Predicted θ")
    if R2_all is not None and np.isfinite(R2_all):
        plt.title(f"{title} | R²={float(R2_all):.4f}")
    else:
        plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    if y_true.size < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot < 1e-12:
        return np.nan
    return 1.0 - ss_res/ss_tot


# ---------------- Core: load model ----------------
def load_model():
    model = EnhancedPINN(hidden=[128,256,256,128], activation="silu", out_temp=8.0, n_depth_freqs=6).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print("Loaded checkpoint:", CKPT_PATH)
    return model


# ---------------- MAIN: prediction-only ----------------
def main_prediction_only():
    forcing = read_forcing_excel(PARAM_FILE)
    time_s = forcing["time_s"].astype(np.float32)

    soil_tag = parse_soil_from_filename(PARAM_FILE, default="CP")
    slope_deg = parse_slope_from_filename(PARAM_FILE, default=0.0)
    if forcing.get("slope_series") is not None:
        sser = forcing["slope_series"]
        if isinstance(sser, np.ndarray) and len(sser) == len(time_s):
            slope_deg = float(np.nanmedian(sser))

    print("Prediction-only | Soil:", soil_tag, "| Slope(deg):", slope_deg)

    depths_pred_m = make_depth_grid_custom(PRED_MAX_DEPTH_M)
    theta0_pred = theta0_profile_constant(depths_pred_m, theta0=PRED_THETA0_CONST)

    model = load_model()

    X = build_features_20(
        time_s=time_s,
        depths_m=depths_pred_m,
        soil_tag=soil_tag,
        slope_deg=slope_deg,
        forcing=forcing,
        theta0_by_depth=theta0_pred,
        soil_thickness_m=THICKNESS_TRAIN_NORM
    )

    with torch.no_grad():
        Y = model(torch.from_numpy(X).to(DEVICE)).cpu().numpy().astype(np.float32)
    Y = Y.reshape(len(depths_pred_m), len(time_s))

    # Save prediction grid
    df_pred = pd.DataFrame(Y, index=depths_pred_m, columns=time_s)
    df_pred.index.name = "Depth_m"
    df_pred.columns.name = "Time_s"
    pred_csv = os.path.join(OUT_DIR, "pred_grid_prediction_only.csv")
    df_pred.to_csv(pred_csv)
    print("Saved:", pred_csv)

    # ALL plots (prediction-only)
    plot_profiles_multiple_times(
        time_s=time_s,
        depths_pred_m=depths_pred_m,
        pred_grid=Y,
        custom_times_hr=CUSTOM_TIMES_HR,
        title="Prediction-only profiles at selected times (0 m at top)"
    )
    plot_heatmap_predicted(time_s, depths_pred_m, Y, title="Predicted θ heatmap (prediction-only)")

    return pred_csv


# ---------------- MAIN: validation (no obs interpolation) ----------------
def main_with_validation():
    forcing = read_forcing_excel(PARAM_FILE)
    monitor = read_monitoring_excel(MONITOR_FILE)

    tF = forcing["time_s"]
    tM = monitor["time_s"]
    idx_map = np.array([int(np.argmin(np.abs(tF - tm))) for tm in tM], dtype=int)

    # timeline = monitoring time (evaluation timeline)
    time_s = tM.copy()

    # align forcing to monitoring time by nearest index (no obs interpolation)
    forcing_aligned = {
        k: (forcing[k][idx_map] if isinstance(forcing.get(k), np.ndarray) and forcing[k].ndim==1 and len(forcing[k])==len(tF) else forcing.get(k))
        for k in forcing.keys()
    }
    for k in ["rain_mm10","temp_C","RH_pct","wind_ms","wind_dir_deg","FVC","ISA","aspect_deg"]:
        forcing_aligned[k] = np.asarray(forcing_aligned[k], dtype=np.float32)

    soil_tag = parse_soil_from_filename(MONITOR_FILE, default="CP")
    slope_deg = parse_slope_from_filename(MONITOR_FILE, default=0.0)
    if forcing.get("slope_series") is not None:
        sser = forcing["slope_series"]
        if isinstance(sser, np.ndarray) and len(sser)==len(tF):
            slope_deg = float(np.nanmedian(sser[idx_map]))

    print("Validation | Soil:", soil_tag, "| Slope(deg):", slope_deg)

    depths_obs_cm = monitor["depths_cm"]
    depths_obs_m = depths_obs_cm / 100.0
    max_depth_m = float(np.nanmax(depths_obs_m))
    depths_pred_m = make_depth_grid_m(max_depth_m)

    theta_obs = monitor["theta_obs"]  # (nZobs, T)
    theta0_obs_by_depth = theta_obs[:, 0].astype(np.float32)

    # theta0 at pred depths: nearest monitored depth (NOT used for validation interpolation)
    theta0_pred = np.zeros_like(depths_pred_m, dtype=np.float32)
    for i, zp in enumerate(depths_pred_m):
        j = int(np.argmin(np.abs(depths_obs_m - zp)))
        theta0_pred[i] = float(theta0_obs_by_depth[j])

    model = load_model()

    X = build_features_20(
        time_s=time_s,
        depths_m=depths_pred_m,
        soil_tag=soil_tag,
        slope_deg=slope_deg,
        forcing=forcing_aligned,
        theta0_by_depth=theta0_pred,
        soil_thickness_m=THICKNESS_TRAIN_NORM
    )

    with torch.no_grad():
        Y = model(torch.from_numpy(X).to(DEVICE)).cpu().numpy().astype(np.float32)
    Y = Y.reshape(len(depths_pred_m), len(time_s))

    # Save dense prediction
    df_pred = pd.DataFrame(Y, index=depths_pred_m, columns=time_s)
    df_pred.index.name = "Depth_m"
    df_pred.columns.name = "Time_s"
    pred_csv = os.path.join(OUT_DIR, "pred_grid_dense_depth.csv")
    df_pred.to_csv(pred_csv)
    print("Saved:", pred_csv)

    # Compare ONLY at monitored depths (nearest pred depth; no interpolation)
    pred_at_obs = np.zeros_like(theta_obs, dtype=np.float32)
    for i, zobs in enumerate(depths_obs_m):
        k = int(np.argmin(np.abs(depths_pred_m - zobs)))
        pred_at_obs[i, :] = Y[k, :]

    # Long table for scatter + metrics
    rows = []
    for i, dcm in enumerate(depths_obs_cm):
        for ti, ts in enumerate(time_s):
            rows.append({
                "depth_cm": float(dcm),
                "time_s": float(ts),
                "theta_obs": float(theta_obs[i, ti]),
                "theta_pred": float(pred_at_obs[i, ti]),
            })
    df_cmp = pd.DataFrame(rows)
    cmp_csv = os.path.join(OUT_DIR, "compare_points.csv")
    df_cmp.to_csv(cmp_csv, index=False)
    print("Saved:", cmp_csv)

    R2_all = r2_score(df_cmp["theta_obs"].values, df_cmp["theta_pred"].values)
    print("R2 (all monitored points):", R2_all)

    # ALL plots (validation)
    plot_profiles_with_obs_multiple_times(
        time_s=time_s,
        depths_pred_m=depths_pred_m,
        pred_grid=Y,
        depths_obs_m=depths_obs_m,
        theta_obs=theta_obs,
        custom_times_hr=CUSTOM_TIMES_HR,
        title=f"Profile comparison at selected times (0 m at top) | R²={float(R2_all):.4f}"
    )

    plot_timeseries_each_depth_with_rain(
    time_s=time_s,
    rain_input=forcing_aligned["rain_mm10"],   # use EXACT unit from PARAM file
    depths_obs_cm=depths_obs_cm,
    theta_obs=theta_obs,
    pred_at_obs=pred_at_obs,
    title_prefix="Time series at depth",
    rain_label="Rain (mm/10min)"
    )

    plot_scatter_obs_vs_pred(
        df_cmp=df_cmp,
        R2_all=R2_all,
        title="Observed vs Predicted (all monitored points)"
    )

    plot_heatmap_predicted(time_s, depths_pred_m, Y, title="Predicted θ heatmap (validation run)")

    return pred_csv, cmp_csv


# ---------------- RUN ----------------
if __name__ == "__main__":
    if isinstance(MONITOR_FILE, str) and len(MONITOR_FILE.strip()) > 0:
        main_with_validation()
    else:
        main_prediction_only()
