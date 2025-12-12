from pathlib import Path
import polars as pl
import os

def make_folder_name(
    N_LANES,
    N_SPEEDS,
    N_DIRECTIONS,
    MASS,
    g,
    tau_h,
    tau_v,
    mu,
    Power,
    F_X_neg_scale,
    F_X_scale,
    F_Y_scale,
    limit_trust,
    control_trust
) -> str:
    return (
        "TESTRUN"
        f"_L{N_LANES}"
        f"_S{N_SPEEDS}"
        f"_D{N_DIRECTIONS}"
        f"_M{MASS}"
        f"_g{g}"
        f"_th{tau_h}"
        f"_tv{tau_v}"
        f"_mu{mu}"
        f"_P{Power}"
        f"_Fxns{F_X_neg_scale}"
        f"_Fxs{F_X_scale}"
        f"_Fys{F_Y_scale}"
        f"limit_trust{limit_trust}"
        f"control_trust{control_trust}"

    )


def init_test_run_folder(config, base_directory):
    folder_name = make_folder_name(
        config.gc.N_LANES,
        config.gc.N_SPEEDS,
        config.gc.N_DIRECTIONS,
        config.pc.M,
        config.pc.g,
        config.pc.tau_h,
        config.pc.tau_v,
        config.pc.mu,
        config.pc.Power,
        config.pc.F_X_neg_scale,
        config.pc.F_X_scale,
        config.pc.F_Y_scale,
        config.pc.limit_trust,
        config.pc.control_trust
    )
    run_path = Path(base_directory) / folder_name
    has_run = os.path.isdir(run_path)
    if has_run:
        return run_path, False
    run_path.mkdir(parents=True, exist_ok=True)
    Setting = pl.DataFrame(
        {
            "N_LANES": [config.gc.N_LANES],
            "N_SPEEDS": [config.gc.N_SPEEDS],
            "N_DIRECTIONS": [config.gc.N_DIRECTIONS],
            "MASS": [config.pc.M],
            "g": [config.pc.g],
            "tau_h": [config.pc.tau_h],
            "tau_v": [config.pc.tau_v],
            "mu": [config.pc.mu],
            "Power": [config.pc.Power],
            "F_X_neg_scale": [config.pc.F_X_neg_scale],
            "F_X_scale": [config.pc.F_X_scale],
            "F_Y_scale": [config.pc.F_Y_scale],
            "limit_trust": [config.pc.limit_trust],
            "control_trust": [config.pc.control_trust]
        }
    )
    Setting.write_parquet(run_path / "Setting.parquet")
    return run_path, True


def save_frame_to_run_dir(
    df: pl.DataFrame,
    run_path,
    name: str
) -> Path:
    file_path = run_path / f"{name}.parquet"
    df.write_parquet(file_path)
    return file_path