import polars as pl
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import os
from pathlib import Path

import Utils.config as configs

class Grid_Sampling:
    
    @staticmethod
    def resample_track_spline(
        df: pl.DataFrame,
        s_step: float = 1.0,
        x_col: str = "x_m",
        y_col: str = "y_m",
        width_cols: tuple[str, str] = ("width_left_m", "width_right_m"),
    ) -> pl.DataFrame:
        """
        Take a track polyline in df[x_col], df[y_col] (approx 5 m spacing)
        and return a smooth 1 m resolution track using a periodic cubic spline.
        """
        # 1) Extract coordinates
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        w_arrays = [df[c].to_numpy() for c in width_cols]

        # 2) Ensure the loop is explicitly closed (first point == last point)
        #    If not, append the first point to the end.
        if np.hypot(x[0] - x[-1], y[0] - y[-1]) > 1e-6:
            x = np.concatenate([x, x[:1]])
            y = np.concatenate([y, y[:1]])
            w_arrays = [np.concatenate([w, w[:1]]) for w in w_arrays]

        # 3) Build arc-length parameter s
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.hypot(dx, dy)
        s = np.concatenate([[0.0], np.cumsum(ds)])  # s[0] = 0, s[-1] = total length
        total_length = s[-1]

        # 4) Fit periodic cubic splines x(s), y(s)
        #    bc_type="periodic" enforces continuity of position and first derivatives
        spline_x = CubicSpline(s, x, bc_type="periodic")
        spline_y = CubicSpline(s, y, bc_type="periodic")

        # 5) Sample every 1 m (or s_step)
        s_new = np.arange(0.0, total_length, s_step)
        x_new = spline_x(s_new)
        y_new = spline_y(s_new)

        # 8) Linearly interpolate width columns along s
        width_new_cols = {}
        for col_name, w in zip(width_cols, w_arrays):
            # 1D linear interpolation in s
            interp = interp1d(
                s,
                w,
                kind="linear",
                fill_value="extrapolate",  # endpoints are essentially periodic anyway
                assume_sorted=True,
            )
            width_new_cols[col_name] = interp(s_new)

        # 6) Return as a new Polars frame
        data = {
            "s_m": s_new,        # arc length along track
            "x_m": x_new,
            "y_m": y_new,
        }
        data.update(width_new_cols)
        return pl.DataFrame(data)

    @staticmethod
    def get_curvature(n1, n2):
        x0 = pl.col(n1).shift(-1)
        x1 = pl.col(n1)
        x2 = pl.col(n1).shift(1)
        y0 = pl.col(n2).shift(-1)
        y1 = pl.col(n2)
        y2 = pl.col(n2).shift(1)
        eps=1e-12
        a = ((x1 - x2)**2 + (y1 - y2)**2).sqrt()
        b = ((x0 - x2)**2 + (y0 - y2)**2).sqrt()
        c = ((x0 - x1)**2 + (y0 - y1)**2).sqrt()
        # signed twice-area via cross product
        area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        A = area2.abs() * 0.5
        den = a * b * c
        num = 4.0 * A
        kappa = pl.when(den > eps).then(num / den).otherwise(pl.lit(0.0)).fill_null(0.0)
        return kappa.alias(f"kappa_mid")
    
    @staticmethod
    def smooth_angular_velocity_triangle(
        df: pl.DataFrame,
        distance_col: str = "s_m",      # your distance column name
        omega_col: str = "angular_velocity",   # your angular velocity column name
        window_length: float = 20.0            # +/- window in meters
    ) -> pl.DataFrame:
        """
        Smooth `angular_velocity` with a symmetric triangular kernel in +/- window_length
        (in meters) around each point. Adds a new column f"{omega_col}_smooth".
        """

        # 1) Extract arrays
        s = df[distance_col].to_numpy()
        w = df[omega_col].to_numpy()
        n = len(w)

        if n < 3:
            # Nothing useful to smooth
            return df.with_columns(
                pl.Series(f"{omega_col}_smooth", w)
            )

        # 2) Estimate step size (assumes ~constant spacing along distance)
        ds = np.diff(s)
        mean_ds = np.median(ds)  # robust step size
        if mean_ds <= 0:
            raise ValueError("Distance column must be strictly increasing.")

        # Number of samples to cover +/- window_length
        half_width_n = max(1, int(round(window_length / mean_ds)))

        # 3) Build symmetric triangular kernel in index-space
        # offsets = -half_width_n, ..., 0, ..., +half_width_n
        offsets = np.arange(-half_width_n, half_width_n + 1)
        # Triangle heights: largest at 0, linearly decreasing to edges
        # height(k) = half_width_n + 1 - |k|
        kernel = (half_width_n + 1 - np.abs(offsets)).astype(float)

        # 4) Normalised convolution, with proper edge handling
        # Numerator: convolution of signal with kernel
        num = np.convolve(w, kernel, mode="same")
        # Denominator: convolution of "1" with kernel (to renormalise near edges)
        denom = np.convolve(np.ones_like(w), kernel, mode="same")

        # Avoid divide-by-zero
        denom[denom == 0] = 1.0

        w_smooth = num / denom

        # 5) Attach back to Polars frame
        return df.with_columns(
            pl.Series(f"{omega_col}_smooth", w_smooth)
        )
    
    @staticmethod
    def subsample_by_angular_velocity(
        df: pl.DataFrame,
        dist_col: str = "s_m",
        omega_col: str = "kappa_mid_smooth",
        ds_min: float = 5.0,   # smallest spacing (tight corners)
        ds_max: float = 20.0,  # largest spacing (straights)
        omega_low_q: float = 10.0,   # low angular velocity percentile
        omega_high_q: float = 90.0,  # high angular velocity percentile
    ) -> pl.DataFrame:
        """
        Subsample track where the spacing between kept points depends on |angular_velocity|:
        - high |ω| → spacing ≈ ds_min
        - low  |ω| → spacing ≈ ds_max

        Uses percentiles to avoid outliers dominating the mapping.
        """

        # 1) Extract arrays
        s = df[dist_col].to_numpy()
        omega = df[omega_col].to_numpy()
        n = len(s)
        if n == 0:
            return df

        # 2) Build a mapping |ω| -> desired Δs in [ds_min, ds_max]
        w_abs = np.abs(omega)

        # Robust min / max using quantiles (avoid outliers)
        w_lo = np.percentile(w_abs, omega_low_q)
        w_hi = np.percentile(w_abs, omega_high_q)
        if w_hi <= w_lo:
            # Degenerate: treat as almost constant angular velocity
            w_lo, w_hi = np.min(w_abs), np.max(w_abs)
            if w_hi <= w_lo:
                # Completely flat track: just take uniform spacing at ds_max
                keep_idx = np.arange(0, n, int(round(ds_max)))  # assuming 1m steps
                return df[keep_idx]

        # Normalise |ω| into [0, 1]
        w_norm = np.clip((w_abs - w_lo) / (w_hi - w_lo), 0.0, 1.0)

        # Linear mapping: w_norm = 0 → ds_max, w_norm = 1 → ds_min
        ds_target = ds_max - (ds_max - ds_min) * w_norm  # per-point desired spacing (in meters)

        # 3) Walk along the track and select indices
        keep_indices = [0]
        curr_idx = 0
        curr_s = s[curr_idx]

        while curr_idx < n - 1:
            # desired step in meters from this point
            ds = ds_target[curr_idx]
            target_s = curr_s + ds

            # move forward until we reach or exceed target_s
            # since s is ~1m resolution and increasing, linear search is fine
            j = curr_idx
            while j < n - 1 and s[j] < target_s:
                j += 1

            if j == curr_idx:
                # Safety: move at least 1 index forward
                j += 1
                if j >= n:
                    break

            keep_indices.append(j)
            curr_idx = j
            curr_s = s[j]

        # Ensure we include the very last point
        if keep_indices[-1] != n - 1:
            keep_indices.append(n - 1)

        keep_indices = np.array(keep_indices, dtype=int)

        # 4) Return subsampled frame
        return df[keep_indices]
        
    @staticmethod
    def frame_thining(config:configs.config, df:pl.DataFrame):
        cols = ["w_tr_right_m","w_tr_left_m"]
        starting_len = len(df)
        df_resampled =  Grid_Sampling.resample_track_spline(df, width_cols=cols)
        df_resampled_len = len(df_resampled)
        df_resampled =  df_resampled.with_columns(Grid_Sampling.get_curvature("x_m", "y_m"))
        df_resampled =  Grid_Sampling.smooth_angular_velocity_triangle(df_resampled, omega_col="kappa_mid", window_length=125)
        df_reduced =    Grid_Sampling.subsample_by_angular_velocity(df_resampled, ds_min = 10, ds_max=20, omega_low_q = 20, omega_high_q = 80)
        df_reduced_len = len(df_reduced)
        print("original length:", starting_len, "resampled lenght", df_resampled_len, "df_reduced_len", df_reduced_len)
        return df_reduced 
    