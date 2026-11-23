"""Shared settings: bridge geometry, test protocol and the feature list.

Everything that describes *the experiment* lives here, so the simulator, the
real Pi logger and the analysis all agree.
"""

# ---- bridge geometry (mm) ----
SPAN_MM = 270.0            # distance between the two support pencils
LOAD_X_MM = 135.0          # load position measured from the left support (midspan)
GAUGE_X_MM = 145.0         # strain gauges: 160 mm from end A = 145 mm from left support
MPU_X_MM = 67.5            # MPU6050 at quarter span (82.5 mm from end A)
PLANK_T_MM = 3.0

# ---- test protocol ----
LOAD_STEPS_G = [0, 100, 200, 300, 400, 500, 600]   # loading; unloading runs back down
HOLD_SECONDS = 5           # wait at each step before reading (keeps creep consistent)
TAP_FS_HZ = 1000           # MPU6050 sample rate during the tap test
TAP_SECONDS = 2.0
STAGES = ["D0", "D1", "D2", "D3"]
STAGE_NAMES = {
    "D0": "Healthy",
    "D1": "Hidden crack",
    "D2": "Section loss (left)",
    "D3": "Severe",
}

G = 9.81 / 1000.0          # grams -> newtons

# ---- features the model uses (one value per test cycle) ----
FEATURES = [
    "stiffness_n_per_mm",   # load / sag slope
    "nonlinearity",         # how much the bridge softens at high load
    "hysteresis_mj",        # energy lost between loading and unloading
    "residual_sag_mm",      # sag left over after unloading
    "strain_per_n_l",       # left gauge microstrain per newton
    "strain_per_n_r",       # right gauge microstrain per newton
    "strain_ratio_lr",      # left / right: one-sided damage
    "freq_hz",              # natural frequency from the tap test
    "damping_ratio",        # how fast the tap vibration dies out
    "ae_count_log",         # log(1 + number of crackles)
    "ae_energy_log",        # log(1 + total crackle energy)
]
ENV_COLUMNS = ["temp_c", "rh_pct"]   # weather, used to correct the features
