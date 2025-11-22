Silverstone Simulator
=====================

Overview
--------
This repository contains a lightweight vehicle simulator and planner for the
Silverstone track lane data. The main script `simulate.py`:

- Loads lane coordinates from `TrackLanes.parquet` (columns like `x_lane0`,
  `y_lane0`, `x_lane1`, `y_lane1`, ... or fallback `x_m`, `y_m`).
- Builds a discrete-time dynamic programming (DP) planner to choose a lane at
  each curvilinear sample to approximately minimize lap time.
- Simulates a kinematic bicycle vehicle model using a pure-pursuit lateral
  controller and a simple longitudinal controller.
- Animates the simulation using `matplotlib` and can save output as MP4
  (requires `ffmpeg`) or GIF (Pillow fallback).

Goals
-----
The script is intended to be:

- Understandable: the code contains detailed comments explaining the physics
  and numeric choices.
- Tunable: many control, planner, and safety parameters are exposed via
  command-line arguments so you can experiment.
- Robust: defaults are chosen so the vehicle is likely to complete the
  circuit rather than immediately leaving the track.

How it works (high level)
-------------------------
1. Data loading: lane polylines are read and resampled onto a common
   curvilinear `s` grid so different lanes can be compared directly.

2. DP planning: for each lane and s-sample we compute a curvature-limited
   speed (using a simple lateral acceleration limit). The DP then finds a
   lane-per-s plan that minimizes the time to traverse the track, with a
   tunable penalty for changing lanes.

3. Simulation: a kinematic bicycle model (state: x, y, yaw, v) is integrated
   forward. Steering is computed with Pure Pursuit toward a lookahead point
   on the currently selected lane; longitudinal control is a proportional
   controller with acceleration/braking limits.

4. Safety: a crude track polygon is built from the outermost lanes for a
   point-in-polygon containment test. If the vehicle leaves this polygon the
   sim stops. The sim also detects when the vehicle reaches the end of the
   resampled path and reports a finished lap time.

Running the simulator
---------------------
Install dependencies (preferably in a virtualenv):

```bash
pip install -r requirements.txt
# If you want MP4 output, install ffmpeg on your system, e.g.:
# sudo apt install ffmpeg
```

Basic run (show animation and save default file):

```bash
python simulate.py --parquet TrackLanes.parquet --time 30
```

Save a GIF and don't show interactive window:

```bash
python simulate.py --parquet TrackLanes.parquet --out out.gif --no-show --time 20
```

Save MP4 (requires `ffmpeg`):

```bash
python simulate.py --parquet TrackLanes.parquet --out out.mp4 --fps 20 --no-show --time 30
```

Key command-line parameters
---------------------------
- `--parquet, -p`: Path to the parquet file (default `TrackLanes.parquet`).
- `--lane, -l`: Requested lane index to use as a starting reference (default 5).
- `--speed, -s`: Target cruising speed used by the DP planner and controller
  (m/s). Default `20.0`. If the vehicle runs off-track, try reducing this.
- `--dt`: Simulation timestep in seconds (default `0.05`).
- `--time`: Total simulation time in seconds (default `60.0`).
- `--out, -o`: Optional output animation filename (MP4/GIF). If omitted, the
  script saves `simulation_output.mp4` by default (or GIF fallback).
- `--fps`: Frames per second for saved animation (default `20`).
- `--no-show`: Do not display the interactive matplotlib window.

Tunable controller / planner options (advanced)
-----------------------------------------------
These options affect the vehicle's behavior. The defaults are tuned to be
conservative so the car is more likely to finish a lap.

- `--max-accel`: Maximum allowed acceleration (m/s^2). Default `3.0`.
- `--max-brake`: Maximum braking (positive number, applied as negative accel).
  Default `6.0`.
- `--kp-speed`: Proportional gain for longitudinal control (default `1.0`).
- `--lookahead-base`: Base lookahead distance (meters) for pure pursuit (default `4.0`).
- `--lookahead-velocity`: Additional lookahead per m/s of speed (default `0.6`).
- `--min-lookahead`, `--max-lookahead`: Bounds for adaptive lookahead.
- `--max-steer-rate`: Limit on how fast the steering angle can change (rad/s).
- `--lane-change-penalty`: Penalty (seconds) added per lane change in DP.
  Lower values encourage more lane changes to achieve faster times.
- `--a-lat-max`: Lateral acceleration limit used to compute curvature-limited
  speeds in the DP planner (m/s^2). Default `8.0`.

Tuning advice to finish the track reliably
-----------------------------------------
- If the vehicle leaves the track immediately: reduce `--speed`, increase
  `--lookahead-base`, or lower `--kp-speed` to make longitudinal control
  gentler.
- Increasing `--min-lookahead` or `--lookahead-velocity` generally smooths
  steering and helps avoid oscillations at higher speeds.
- If the DP chooses unrealistic lane changes, increase `--lane-change-penalty`.
- For more aggressive lap times, raise `--speed` and lower `--lane-change-penalty`,
  but do this incrementally and verify the vehicle still stays within the
  track polygon.

Implementation notes (for the curious)
--------------------------------------
- Kinematic bicycle model: simple, efficient, good for low-to-moderate
  speeds; it ignores tire slip and full vehicle dynamics.
- Pure Pursuit: geometric controller that aims a lookahead point on the path.
  It is simple and robust; lookahead tuning is important for stability.
- Dynamic Programming: we discretize the track along arc length and allow
  lane changes between adjacent lanes per step. The cost is the time to
  traverse each segment at a curvature-limited speed plus a penalty for
  changing lanes.

If you want help tuning
-----------------------
Tell me what happened when you ran the simulator (logs, output filenames,
which `--lane` you used). I can:

- Automatically pick safer defaults (e.g., lower starting speed) and re-run
  a short smoke test that saves an animation and shares the resulting file
  path.
- Further refine the DP (finer s-grid, different lane-change costs).
- Replace pure pursuit with a more advanced controller (e.g., Stanley or
  a model-predictive controller) if you want higher performance.

Enjoy experimenting — say which next step you want me to take!
