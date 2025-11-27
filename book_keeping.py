from pathlib import Path
import polars as pl

def make_folder_name(
    N_LANES,
    N_SPEEDS,
    N_DIRECTIONS,
    MASS,
    g,
    tau_h,
    tau_v,
    mu,
    F_MAX_ACC,
    F_X_NEG_SCALE,
    F_X_MAX_SCALE,
    F_Y_MAX_SCALE,
    rho
) -> str:
    # you can tweak this format however you like
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
        f"_Facc{F_MAX_ACC}"
        f"_Fxns{F_X_NEG_SCALE}"
        f"_Fxm{F_X_MAX_SCALE}"
        f"_Fyms{F_Y_MAX_SCALE}"
        f"_rho{rho}"

    )


def init_test_run_folder(
    base_dir,
    N_LANES,
    N_SPEEDS,
    N_DIRECTIONS,
    MASS,
    g,
    tau_h,
    tau_v,
    mu,
    F_MAX_ACC,
    F_X_NEG_SCALE,
    F_X_MAX_SCALE,
    F_Y_MAX_SCALE,
    rho
):
    # 1) build folder name
    folder_name = make_folder_name(
        N_LANES,
        N_SPEEDS,
        N_DIRECTIONS,
        MASS,
        g,
        tau_h,
        tau_v,
        mu,
        F_MAX_ACC,
        F_X_NEG_SCALE,
        F_X_MAX_SCALE,
        F_Y_MAX_SCALE,
        rho
    )

    # 2) create folder
    run_path = Path(base_dir) / folder_name
    run_path.mkdir(parents=True, exist_ok=True)

    # 3) create Setting frame (one row with all arguments)
    Setting = pl.DataFrame(
        {
            "N_LANES": [N_LANES],
            "N_SPEEDS": [N_SPEEDS],
            "N_DIRECTIONS": [N_DIRECTIONS],
            "MASS": [MASS],
            "g": [g],
            "tau_h": [tau_h],
            "tau_v": [tau_v],
            "mu": [mu],
            "F_MAX_ACC": [F_MAX_ACC],
            "F_X_NEG_SCALE": [F_X_NEG_SCALE],
            "F_X_MAX_SCALE": [F_X_MAX_SCALE],
            "F_Y_MAX_SCALE": [F_Y_MAX_SCALE],
            "rho": [rho]
        }
    )

    # 4) save settings into the folder
    Setting.write_parquet(run_path / "Setting.parquet")
    # or CSV:
    # Setting.write_csv(run_path / "Setting.csv")

    return run_path, Setting


def save_frame_to_run_dir(
    df: pl.DataFrame,
    run_path,
    name: str
) -> Path:
    file_path = run_path / f"{name}.parquet"
    df.write_parquet(file_path)
    return file_path