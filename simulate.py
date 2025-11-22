#!/usr/bin/env python3
"""
Silverstone simulator (annotated)

This script loads lane data from a parquet file (expected columns like
`x_lane0`, `y_lane0`, `x_lane1`, `y_lane1`, ... or fallback `x_m`, `y_m`) and
performs a numerical simulation of a simple car model driving around the
track. The simulation contains three main parts:

- Path / lane handling and resampling
- A dynamic-programming (DP) routine to compute a time-optimal lane plan
    (discrete approximation over path distance)
- A forward-time simulation using a kinematic bicycle model with a
    pure-pursuit lateral controller and a simple longitudinal controller

The script also animates the vehicle and can save the animation as an MP4
(via ffmpeg) or GIF (via Pillow) using matplotlib's animation API.

The comments inside the file explain the physics and numerics used so you can
follow the implementation even if you're unfamiliar with vehicle models.
"""

from pathlib import Path
import argparse
import math
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.path import Path as MplPath


def load_lanes(path: Path):
        """Load the parquet file containing lane coordinates and return a Polars DataFrame.

        Practical notes:
        - Parquet is efficient for column-oriented data. Using Polars makes reads fast.
        - We return the raw dataframe and let other helpers interpret which columns
            represent which lane coordinates.
        - This function intentionally does not validate columns beyond letting the
            downstream helpers decide how to handle missing lanes.
        """
        return pl.read_parquet(path)


def lane_xy_arrays(df, lane_index: int):
        """Return numpy arrays (x, y) for a requested lane index.

        The function is defensive:
        - If the requested lane index exists in the dataframe (columns like
            `x_lane5`, `y_lane5`), those arrays are returned.
        - If the exact requested lane does not exist but other `x_lane*` columns
            exist, the numerically closest lane index is chosen. This allows the
            user to request a lane that isn't present without crashing.
        - If no `x_lane*` columns exist, but a centerline `x_m`, `y_m` exists,
            we use that as a fallback.

        Returns:
            - (x_array, y_array): numpy arrays of the same length giving coordinates
                along the lane.
        """
        cols = df.columns
        x_col = f"x_lane{lane_index}"
        y_col = f"y_lane{lane_index}"
        # Direct lookup if exactly present
        if x_col in cols and y_col in cols:
                return np.asarray(df[x_col].to_numpy()), np.asarray(df[y_col].to_numpy())
        # Otherwise pick the numerically closest available lane
        x_cols = [c for c in cols if c.startswith('x_lane')]
        if x_cols:
                # Extract numeric suffixes safely (assumes 'x_lane{n}' format)
                idxs = [int(c.split('lane')[-1]) for c in x_cols]
                closest = min(idxs, key=lambda i: abs(i - lane_index))
                return np.asarray(df[f'x_lane{closest}'].to_numpy()), np.asarray(df[f'y_lane{closest}'].to_numpy())
        # Final fallback to a centerline naming convention if available
        if 'x_m' in cols and 'y_m' in cols:
                return np.asarray(df['x_m'].to_numpy()), np.asarray(df['y_m'].to_numpy())
        # If nothing matches, raise an informative error
        raise RuntimeError('No lane columns found')


def build_path_interp(x, y):
        """Create simple interpolators x(s), y(s) and heading(s) along a polyline.

        Inputs:
            - x, y: 1D arrays describing a path (ordered points along a lane)

        Produces:
            - s: array of cumulative distances along the path (same length as x)
            - x_of_s(qs), y_of_s(qs): linear interpolation of x,y over s
            - heading_of_s(qs): interpolated heading angle (radians) along s

        Notes:
        - We compute headings by taking the arctangent of forward differences.
            This is a discrete approximation of the tangent direction.
        - Linear interpolation is used for simplicity. For smoother paths one
            could use splines, but linear interp is faster and adequate here.
        """
        dx = np.diff(x); dy = np.diff(y)
        ds = np.hypot(dx, dy)
        # cumulative arc-length; first point has s=0
        s = np.concatenate(([0.0], np.cumsum(ds)))

        # Return small helper functions that close over (s, x, y)
        def x_of_s(qs): return np.interp(qs, s, x)
        def y_of_s(qs): return np.interp(qs, s, y)

        # Approximate heading per vertex. We append the last segment heading so
        # the headings array has the same length as x and y.
        headings = np.arctan2(np.concatenate((dy, [dy[-1]])), np.concatenate((dx, [dx[-1]])))
        def heading_of_s(qs): return np.interp(qs, s, headings)

        return s, x_of_s, y_of_s, heading_of_s


def nearest_index_on_path(xp, yp, x_path, y_path):
    """Return the index of the closest vertex on the provided path to (xp, yp).

    This is a fast and simple nearest-vertex method. For higher accuracy you
    could compute the perpendicular projection onto each segment, but that
    is more expensive and seldom necessary for visualization/control.
    """
    d2 = (x_path - xp)**2 + (y_path - yp)**2
    return int(np.argmin(d2))


def kinematic_bicycle_step(state, a, delta, dt, L=2.6):
        """Integrate the kinematic bicycle model for one timestep.

        State vector: [x, y, yaw, v]
            - x, y: planar position of the vehicle reference point
            - yaw: heading (radians)
            - v: longitudinal speed (m/s)

        Inputs:
            - a: longitudinal acceleration command (m/s^2)
            - delta: steering angle command (radians)
            - dt: timestep (s)
            - L: wheelbase length (m)

        The kinematic bicycle uses a single front steering angle and approximates
        the instantaneous rotation rate as v/L * tan(delta). This model is
        commonly used at low-to-moderate speeds where tire slip is small.
        """
        x, y, yaw, v = state
        # Protect against unrealistically large steering inputs
        max_steer = math.radians(45.0)
        delta = max(-max_steer, min(max_steer, delta))

        # Euler integration (simple, efficient). For higher accuracy use RK4
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += v / L * math.tan(delta) * dt
        v += a * dt

        return np.array([x, y, yaw, v])


def pure_pursuit_control(state, path_x, path_y, path_s, lookahead=8.0, L=2.6):
    """Compute steering using the Pure Pursuit geometric controller.

    Pure Pursuit works by choosing a lookahead point on the path a fixed
    curvilinear distance ahead (lookahead) and steering toward it. The
    steering command is chosen so the instantaneous circular arc through the
    vehicle and the lookahead point has radius R = L / (2*sin(alpha)/Ld)
    which leads to the formula used below.

    Advantages: simple and stable for many tracking tasks. Disadvantages:
    - Performance can degrade for very small lookahead distances.
    - It is geometric (doesn't directly consider vehicle dynamics).
    """
    x, y, yaw, v = state
    # Nearest vertex gives an index on the path; for coarse data this is OK
    idx = nearest_index_on_path(x, y, path_x, path_y)
    s_cur = path_s[idx]
    # Target curvilinear coordinate ahead of the current location
    s_t = min(s_cur + lookahead, path_s[-1])
    # Interpolate the lookahead point in world coordinates
    tx = np.interp(s_t, path_s, path_x); ty = np.interp(s_t, path_s, path_y)
    # Relative angle from vehicle heading to the line-of-sight
    alpha = math.atan2(ty - y, tx - x) - yaw
    # Normalize to [-pi, pi]
    while alpha > math.pi: alpha -= 2*math.pi
    while alpha < -math.pi: alpha += 2*math.pi
    if lookahead <= 0: return 0.0
    # Pure pursuit steering law (approximate analytical expression)
    return math.atan2(2 * L * math.sin(alpha), lookahead)


def draw_car_polygon(x, y, yaw, length=4.0, width=2.0):
    """Return a small triangular polygon representing the vehicle for plotting.

    This function is purely cosmetic: it returns three points that approximate
    a car shape in the vehicle frame and transforms them to world coordinates
    using the heading `yaw`.
    """
    corners = np.array([[length*0.6,0.],[ -length*0.4, -width/2.], [-length*0.4, width/2.]])
    R = np.array([[math.cos(yaw), -math.sin(yaw)],[math.sin(yaw), math.cos(yaw)]])
    return (R @ corners.T).T + np.array([x, y])


def compute_dp_lane_plan(lane_paths, lane_list, speed_target, lane_change_penalty=0.3, a_lat_max=8.0):
    """Compute a simple time-minimizing lane plan via discrete DP.

    Details:
    - We choose a base lane (the first in `lane_list`) to define the shared
      curvilinear abscissa `s_dense`. All lanes are resampled onto that s-grid
      so we have comparable indices across lanes.
    - For each lane and s index we estimate curvature kappa using derivatives
      of x(s), y(s). Using a simple lateral acceleration constraint
      `a_lat_max`, we compute a curvature-limited speed vcurv = sqrt(a_lat_max/kappa).
    - The DP cost to move from s_i to s_{i+1} on lane j is the time to traverse
      the segment ds at the lane's allowed speed plus a small penalty if a lane
      change occurs. The DP only allows lane changes between adjacent lanes
      (off = -1, 0, +1).

    This returns a per-s sample lane assignment approximating a time-optimal
    choice under the given assumptions.
    """
    base = lane_list[0]
    base_x, base_y = lane_paths[base]
    s_dense, _, _, _ = build_path_interp(base_x, base_y)
    M = len(s_dense)

    lane_xy = {}
    lane_v = {}
    # Resample lanes and estimate curvature-limited speeds
    for li in lane_list:
        lx, ly = lane_paths[li]
        _, xsi, ysi, _ = build_path_interp(lx, ly)
        xi = xsi(s_dense); yi = ysi(s_dense)
        lane_xy[li] = (xi, yi)
        # Spatial derivatives w.r.t s for curvature computation
        dx = np.gradient(xi, s_dense); dy = np.gradient(yi, s_dense)
        ddx = np.gradient(dx, s_dense); ddy = np.gradient(dy, s_dense)
        # Curvature formula for a 2D parametric curve
        kappa = np.abs(dx*ddy - dy*ddx) / np.maximum((dx*dx + dy*dy)**1.5, 1e-9)
        # Avoid dividing by zero curvature: enforce small floor
        vcurv = np.sqrt(np.maximum(a_lat_max / np.maximum(kappa, 1e-9), 0.0))
        lane_v[li] = np.minimum(speed_target, vcurv)

    seg_ds = np.diff(s_dense)
    L = len(lane_list)
    INF = 1e12
    # dp[i, j] = minimal time cost to reach s index i while being on lane j
    dp = np.full((M, L), INF); pred = np.full((M, L), -1, dtype=int)
    dp[0, :] = INF; dp[0, 0] = 0.0

    # Forward propagate costs along s
    for i in range(M-1):
        ds = seg_ds[i]
        for j in range(L):
            if dp[i,j] >= INF: continue
            # Consider staying or switching to adjacent lanes
            for off in (0,-1,1):
                nj = j+off
                if nj<0 or nj>=L: continue
                li = lane_list[nj]
                v_allowed = lane_v[li][i]
                if v_allowed < 0.1: v_allowed = 0.1
                dt_seg = ds / v_allowed
                cost = dp[i,j] + dt_seg + abs(nj-j)*lane_change_penalty
                if cost < dp[i+1,nj]: dp[i+1,nj]=cost; pred[i+1,nj]=j

    # Backtrack to obtain lane index per s sample
    end = int(np.argmin(dp[-1,:]))
    plan = np.zeros(M, dtype=int); plan[-1]=end
    for i in range(M-1,0,-1): plan[i-1]=pred[i,plan[i]]
    lane_plan = [lane_list[c] for c in plan]
    return s_dense, lane_xy, lane_plan


def simulate_and_animate(
    parquet_path: Path,
    lane_index=5,
    speed_target=20.0,
    dt=0.05,
    sim_time=60.0,
    out_path: Path | None = None,
    fps: int = 20,
    show: bool = True,
    # control tuning parameters (safer defaults to help finish the track)
    max_accel: float = 3.0,
    max_brake: float = 6.0,
    kp_speed: float = 1.0,
    lookahead_base: float = 4.0,
    lookahead_vel: float = 0.6,
    min_lookahead: float = 4.0,
    max_lookahead: float = 25.0,
    max_steer_rate: float = 1.0,
    lane_change_penalty: float = 0.25,
    a_lat_max: float = 8.0,
):
    """Run the lane DP planner, simulate the vehicle, and animate/save output.

    The function orchestrates data loading, planning, forward simulation, and
    animation. It is intended as the top-level routine that the CLI invokes.
    """
    # 1) Load lane data and detect available lane indices
    df = load_lanes(parquet_path)
    cols = df.columns
    lane_indices = sorted([int(c.split('lane')[-1]) for c in cols if c.startswith('x_lane') and c.split('lane')[-1].isdigit()])
    if not lane_indices:
        # If there are no explicit lane columns, try to use the requested lane
        x0,y0 = lane_xy_arrays(df,lane_index); lane_indices=[lane_index]; lane_paths={lane_index:(x0,y0)}
    else:
        # Build a dict mapping lane index -> (x_array, y_array)
        lane_paths = {i: lane_xy_arrays(df,i) for i in lane_indices}

    # 2) Compute a global lane plan using DP (a per-sample lane assignment)
    #    Pass tuning parameters into the planner so curvature-limited speeds
    #    and lane-change penalty are respected.
    s_dense, lane_xy_resampled, lane_plan = compute_dp_lane_plan(
        lane_paths, lane_indices, speed_target, lane_change_penalty=lane_change_penalty, a_lat_max=a_lat_max
    )

    # 3) Build a robust track polygon for containment tests.
    #    Instead of assuming lane ordering, compute a convex hull over all
    #    lane points. A convex hull gives a simple outer polygon that fully
    #    contains the lanes and avoids self-intersections that broke earlier
    #    simplistic heuristics.
    all_pts = []
    for li in lane_indices:
        lx, ly = lane_paths[li]
        all_pts.extend(list(zip(lx, ly)))
    all_pts = np.asarray(all_pts)

    # Monotone chain (Andrew) convex hull implementation (returns CCW hull)
    def convex_hull(points):
        pts = sorted(map(tuple, points))
        if len(pts) <= 2:
            return pts
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        # Concatenate lower and upper to get full hull; omit last point of each
        hull = lower[:-1] + upper[:-1]
        return hull

    hull_points = convex_hull(all_pts)
    poly_points = hull_points
    track_poly = MplPath(poly_points)

    # 4) Initialize simulation state using the first point on the chosen lane
    chosen_lane = lane_plan[0]
    path_x, path_y = lane_xy_resampled[chosen_lane]
    s, x_of_s, y_of_s, heading_of_s = build_path_interp(path_x, path_y)
    # state = [x, y, yaw, v]
    # Start with a small initial forward speed to avoid overshoot from zero
    state = np.array([path_x[0], path_y[0], heading_of_s(0), 1.0])

    # 5) Setup plotting / animation helpers
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_aspect('equal', adjustable='datalim')
    for i in lane_indices:
        lx,ly = lane_paths[i]; ax.plot(lx,ly,color='gray',linewidth=0.6)

    line_car, = ax.plot([],[],'-k')
    car_poly=None; time_text=ax.text(0.02,0.95,'',transform=ax.transAxes)
    path_plot, = ax.plot(path_x, path_y, color='blue', linewidth=0.8, alpha=0.8)
    poly_x=[p[0] for p in poly_points]+[poly_points[0][0]]; poly_y=[p[1] for p in poly_points]+[poly_points[0][1]]
    ax.plot(poly_x, poly_y, color='black', linewidth=0.6, alpha=0.4)

    # Simulation bookkeeping
    sim_steps = int(sim_time/dt)
    history_x=[]; history_y=[]
    # base_x/base_y are the resampled coordinates corresponding to the DP's s-grid
    base_x = lane_xy_resampled[lane_indices[0]][0]; base_y = lane_xy_resampled[lane_indices[0]][1]
    # steering smoothing state and stop flag to avoid repeated messages
    prev_delta = 0.0
    stopped = False

    def init():
        # Initialize axis limits and empty artists
        ax.set_xlim(min(poly_x)-20, max(poly_x)+20); ax.set_ylim(min(poly_y)-20, max(poly_y)+20)
        line_car.set_data([],[]); time_text.set_text(''); return line_car, time_text

    ani_ref={'ani':None}

    def update(frame):
        """Advance simulation by one timestep and update visualization.

        The update loop implements a simple control stack:
          - Longitudinal: P controller pushing speed toward `speed_target`.
          - Lateral: Pure pursuit controller following the currently-selected lane.
          - Integrate vehicle using the kinematic bicycle model.
        """
        nonlocal state, car_poly, chosen_lane, path_x, path_y, s, prev_delta, stopped
        t = frame*dt
        # Longitudinal controller: proportional with acceleration/brake limits
        raw_a = kp_speed * (speed_target - state[3])
        a = max(-max_brake, min(max_accel, raw_a))

        # Determine current s-index using the DP base path (nearest vertex)
        idx_s = nearest_index_on_path(state[0], state[1], base_x, base_y)
        desired = lane_plan[idx_s]
        if desired != chosen_lane:
            # Switch the controller's reference path to the newly chosen lane
            chosen_lane = desired; path_x, path_y = lane_xy_resampled[chosen_lane]
            s, x_of_s, y_of_s, heading_of_s = build_path_interp(path_x, path_y)
            path_plot.set_data(path_x, path_y)

        # Adaptive lookahead (in meters): grows with speed but clipped
        lookahead = lookahead_base + lookahead_vel * state[3]
        lookahead = max(min_lookahead, min(max_lookahead, lookahead))
        # Compute raw steering via pure pursuit
        delta_cmd = pure_pursuit_control(state, path_x, path_y, s_dense, lookahead=lookahead)
        # Smooth steering by limiting the rate of change of steering angle
        max_delta_change = max_steer_rate * dt
        delta = prev_delta
        if delta_cmd > prev_delta:
            delta = min(prev_delta + max_delta_change, delta_cmd)
        else:
            delta = max(prev_delta - max_delta_change, delta_cmd)
        prev_delta = delta
        # Integrate vehicle one step using the kinematic bicycle model
        state = kinematic_bicycle_step(state, a, delta, dt)

        # Update history and drawn car polygon
        history_x.append(state[0]); history_y.append(state[1])
        line_car.set_data(history_x, history_y)
        if car_poly is not None: car_poly.remove()
        pts = draw_car_polygon(state[0], state[1], state[2]); carpoly = ax.fill(pts[:,0], pts[:,1], color='red', zorder=5)
        car_poly = carpoly[0]
        time_text.set_text(f"t={t:.1f}s v={state[3]:.1f} m/s lane={chosen_lane}")

        # If the vehicle leaves the track polygon (our crude containment test), stop once
        if not stopped and not track_poly.contains_point((state[0], state[1])):
            stopped = True
            print(f"Car left the track at t={t:.2f}s (lane {chosen_lane}). Stopping.")
            try: ani_ref['ani'].event_source.stop()
            except Exception: pass
        # If we reach the end of the base path's s-grid, consider the lap finished
        if not stopped and idx_s >= len(base_x)-2:
            stopped = True
            print(f"Finished track at t={t:.2f}s. Stopping.")
            try: ani_ref['ani'].event_source.stop()
            except Exception: pass
        return line_car, car_poly, time_text

    ani = animation.FuncAnimation(fig, update, frames=sim_steps, init_func=init, interval=dt*1000, blit=False)
    ani_ref['ani']=ani

    # Save the animation if requested. Prefer MP4 with ffmpeg; fallback to GIF.
    if out_path is not None:
        outp = Path(out_path); suffix = outp.suffix.lower(); print(f"Saving to {outp} (fps={fps})")
        try:
            if suffix=='.gif':
                writer = PillowWriter(fps=fps); ani.save(str(outp), writer=writer)
            else:
                # Some matplotlib versions expose writers via a registry mapping
                try:
                    Writer = animation.writers['ffmpeg']
                except Exception:
                    Writer = None
                if Writer is None:
                    # If ffmpeg is not available in the environment, fall back to GIF
                    gif_out = outp.with_suffix('.gif')
                    print('ffmpeg unavailable; falling back to GIF:', gif_out)
                    writer = PillowWriter(fps=fps)
                    ani.save(str(gif_out), writer=writer)
                else:
                    writer = Writer(fps=fps, metadata=dict(artist='simulate.py'))
                    ani.save(str(outp), writer=writer)
        except Exception as e:
            import traceback
            print('Save error:', e)
            traceback.print_exc()

    if show: plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet','-p',default='TrackLanes.parquet')
    p.add_argument('--lane','-l',type=int,default=5)
    p.add_argument('--speed','-s',type=float,default=20.0)
    p.add_argument('--dt',type=float,default=0.05)
    p.add_argument('--time',type=float,default=60.0)
    p.add_argument('--out','-o',default=None)
    p.add_argument('--fps',type=int,default=20)
    p.add_argument('--no-show',action='store_true')
    args = p.parse_args()
    # If user didn't provide --out, save to a default file in the workspace
    if args.out:
        out_arg = Path(args.out)
    else:
        out_arg = Path('simulation_output.mp4')
        print(f"--out not provided; saving animation to {out_arg}")

    simulate_and_animate(Path(args.parquet), lane_index=args.lane, speed_target=args.speed, dt=args.dt, sim_time=args.time, out_path=out_arg, fps=args.fps, show=(not args.no_show))


if __name__=='__main__':
    main()
