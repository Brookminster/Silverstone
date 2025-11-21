#!/usr/bin/env python3
"""Clean Silverstone simulator (single-file) with DP lane planning and animation saving.

Use this file if `simulate.py` is corrupted. It reads `TrackLanes.parquet` and
produces an MP4/GIF if `--out` is specified.
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
    return pl.read_parquet(path)


def lane_xy_arrays(df, lane_index: int):
    cols = df.columns
    x_col = f"x_lane{lane_index}"
    y_col = f"y_lane{lane_index}"
    if x_col in cols and y_col in cols:
        return np.asarray(df[x_col].to_numpy()), np.asarray(df[y_col].to_numpy())
    x_cols = [c for c in cols if c.startswith('x_lane')]
    if x_cols:
        idxs = [int(c.split('lane')[-1]) for c in x_cols]
        closest = min(idxs, key=lambda i: abs(i - lane_index))
        return np.asarray(df[f'x_lane{closest}'].to_numpy()), np.asarray(df[f'y_lane{closest}'].to_numpy())
    if 'x_m' in cols and 'y_m' in cols:
        return np.asarray(df['x_m'].to_numpy()), np.asarray(df['y_m'].to_numpy())
    raise RuntimeError('No lane columns found')


def build_path_interp(x, y):
    dx = np.diff(x); dy = np.diff(y)
    ds = np.hypot(dx, dy)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    def x_of_s(qs): return np.interp(qs, s, x)
    def y_of_s(qs): return np.interp(qs, s, y)
    headings = np.arctan2(np.concatenate((dy, [dy[-1]])), np.concatenate((dx, [dx[-1]])))
    def heading_of_s(qs): return np.interp(qs, s, headings)
    return s, x_of_s, y_of_s, heading_of_s


def nearest_index_on_path(xp, yp, x_path, y_path):
    d2 = (x_path - xp)**2 + (y_path - yp)**2
    return int(np.argmin(d2))


def kinematic_bicycle_step(state, a, delta, dt, L=2.6):
    x, y, yaw, v = state
    max_steer = math.radians(45.0)
    delta = max(-max_steer, min(max_steer, delta))
    x += v * math.cos(yaw) * dt
    y += v * math.sin(yaw) * dt
    yaw += v / L * math.tan(delta) * dt
    v += a * dt
    return np.array([x, y, yaw, v])


def pure_pursuit_control(state, path_x, path_y, path_s, lookahead=8.0, L=2.6):
    x, y, yaw, v = state
    idx = nearest_index_on_path(x, y, path_x, path_y)
    s_cur = path_s[idx]
    s_t = min(s_cur + lookahead, path_s[-1])
    tx = np.interp(s_t, path_s, path_x); ty = np.interp(s_t, path_s, path_y)
    alpha = math.atan2(ty - y, tx - x) - yaw
    while alpha > math.pi: alpha -= 2*math.pi
    while alpha < -math.pi: alpha += 2*math.pi
    if lookahead <= 0: return 0.0
    return math.atan2(2 * L * math.sin(alpha), lookahead)


def draw_car_polygon(x, y, yaw, length=4.0, width=2.0):
    corners = np.array([[length*0.6,0.],[ -length*0.4, -width/2.], [-length*0.4, width/2.]])
    R = np.array([[math.cos(yaw), -math.sin(yaw)],[math.sin(yaw), math.cos(yaw)]])
    return (R @ corners.T).T + np.array([x, y])


def compute_dp_lane_plan(lane_paths, lane_list, speed_target, lane_change_penalty=0.3, a_lat_max=8.0):
    base = lane_list[0]
    base_x, base_y = lane_paths[base]
    s_dense, _, _, _ = build_path_interp(base_x, base_y)
    M = len(s_dense)
    lane_xy = {}
    lane_v = {}
    for li in lane_list:
        lx, ly = lane_paths[li]
        _, xsi, ysi, _ = build_path_interp(lx, ly)
        xi = xsi(s_dense); yi = ysi(s_dense)
        lane_xy[li] = (xi, yi)
        dx = np.gradient(xi, s_dense); dy = np.gradient(yi, s_dense)
        ddx = np.gradient(dx, s_dense); ddy = np.gradient(dy, s_dense)
        kappa = np.abs(dx*ddy - dy*ddx) / np.maximum((dx*dx + dy*dy)**1.5, 1e-9)
        vcurv = np.sqrt(np.maximum(a_lat_max / np.maximum(kappa, 1e-9), 0.0))
        lane_v[li] = np.minimum(speed_target, vcurv)

    seg_ds = np.diff(s_dense)
    L = len(lane_list)
    INF = 1e12
    dp = np.full((M, L), INF); pred = np.full((M, L), -1, dtype=int)
    dp[0, :] = INF; dp[0, 0] = 0.0
    for i in range(M-1):
        ds = seg_ds[i]
        for j in range(L):
            if dp[i,j] >= INF: continue
            for off in (0,-1,1):
                nj = j+off
                if nj<0 or nj>=L: continue
                li = lane_list[nj]
                v_allowed = lane_v[li][i]
                if v_allowed < 0.1: v_allowed = 0.1
                dt_seg = ds / v_allowed
                cost = dp[i,j] + dt_seg + abs(nj-j)*lane_change_penalty
                if cost < dp[i+1,nj]: dp[i+1,nj]=cost; pred[i+1,nj]=j

    end = int(np.argmin(dp[-1,:]))
    plan = np.zeros(M, dtype=int); plan[-1]=end
    for i in range(M-1,0,-1): plan[i-1]=pred[i,plan[i]]
    lane_plan = [lane_list[c] for c in plan]
    return s_dense, lane_xy, lane_plan


def simulate_and_animate(parquet_path: Path, lane_index=5, speed_target=20.0, dt=0.05, sim_time=60.0, out_path: Path | None = None, fps: int = 20, show: bool = True):
    df = load_lanes(parquet_path)
    cols = df.columns
    lane_indices = sorted([int(c.split('lane')[-1]) for c in cols if c.startswith('x_lane') and c.split('lane')[-1].isdigit()])
    if not lane_indices:
        x0,y0 = lane_xy_arrays(df,lane_index); lane_indices=[lane_index]; lane_paths={lane_index:(x0,y0)}
    else:
        lane_paths = {i: lane_xy_arrays(df,i) for i in lane_indices}

    s_dense, lane_xy_resampled, lane_plan = compute_dp_lane_plan(lane_paths, lane_indices, speed_target)

    min_lane = min(lane_indices); max_lane = max(lane_indices)
    left_x,left_y = lane_paths[min_lane]; right_x,right_y = lane_paths[max_lane]
    poly_points = list(zip(right_x,right_y)) + list(zip(left_x[::-1], left_y[::-1]))
    track_poly = MplPath(poly_points)

    chosen_lane = lane_plan[0]
    path_x, path_y = lane_xy_resampled[chosen_lane]
    s, x_of_s, y_of_s, heading_of_s = build_path_interp(path_x, path_y)
    state = np.array([path_x[0], path_y[0], heading_of_s(0), 0.0])

    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_aspect('equal', adjustable='datalim')
    for i in lane_indices:
        lx,ly = lane_paths[i]; ax.plot(lx,ly,color='gray',linewidth=0.6)

    line_car, = ax.plot([],[],'-k')
    car_poly=None; time_text=ax.text(0.02,0.95,'',transform=ax.transAxes)
    path_plot, = ax.plot(path_x, path_y, color='blue', linewidth=0.8, alpha=0.8)
    poly_x=[p[0] for p in poly_points]+[poly_points[0][0]]; poly_y=[p[1] for p in poly_points]+[poly_points[0][1]]
    ax.plot(poly_x, poly_y, color='black', linewidth=0.6, alpha=0.4)

    sim_steps = int(sim_time/dt)
    history_x=[]; history_y=[]
    base_x = lane_xy_resampled[lane_indices[0]][0]; base_y = lane_xy_resampled[lane_indices[0]][1]

    def init():
        ax.set_xlim(min(poly_x)-20, max(poly_x)+20); ax.set_ylim(min(poly_y)-20, max(poly_y)+20)
        line_car.set_data([],[]); time_text.set_text(''); return line_car, time_text

    ani_ref={'ani':None}

    def update(frame):
        nonlocal state, car_poly, chosen_lane, path_x, path_y, s
        t = frame*dt
        a = 1.0*(speed_target-state[3])
        idx_s = nearest_index_on_path(state[0], state[1], base_x, base_y)
        desired = lane_plan[idx_s]
        if desired != chosen_lane:
            chosen_lane = desired; path_x, path_y = lane_xy_resampled[chosen_lane]
            s, x_of_s, y_of_s, heading_of_s = build_path_interp(path_x, path_y)
            path_plot.set_data(path_x, path_y)
        delta = pure_pursuit_control(state, path_x, path_y, s_dense, lookahead=max(6.0, state[3]*0.8))
        state = kinematic_bicycle_step(state, a, delta, dt)
        history_x.append(state[0]); history_y.append(state[1])
        line_car.set_data(history_x, history_y)
        if car_poly is not None: car_poly.remove()
        pts = draw_car_polygon(state[0], state[1], state[2]); carpoly = ax.fill(pts[:,0], pts[:,1], color='red', zorder=5)
        car_poly = carpoly[0]
        time_text.set_text(f"t={t:.1f}s v={state[3]:.1f} m/s lane={chosen_lane}")
        if not track_poly.contains_point((state[0], state[1])):
            print(f"Car left the track at t={t:.2f}s (lane {chosen_lane}). Stopping.")
            try: ani_ref['ani'].event_source.stop()
            except Exception: pass
        return line_car, car_poly, time_text

    ani = animation.FuncAnimation(fig, update, frames=sim_steps, init_func=init, interval=dt*1000, blit=False)
    ani_ref['ani']=ani

    if out_path is not None:
        outp = Path(out_path); suffix = outp.suffix.lower(); print(f"Saving to {outp} (fps={fps})")
        try:
            if suffix=='.gif':
                writer = PillowWriter(fps=fps); ani.save(str(outp), writer=writer)
            else:
                # Access the registry via indexing — some matplotlib versions
                # expose MovieWriterRegistry without a `.get` method.
                try:
                    Writer = animation.writers['ffmpeg']
                except Exception:
                    Writer = None
                if Writer is None:
                    # Fall back to GIF if ffmpeg writer not available
                    gif_out = outp.with_suffix('.gif')
                    print('ffmpeg unavailable; falling back to GIF:', gif_out)
                    writer = PillowWriter(fps=fps)
                    ani.save(str(gif_out), writer=writer)
                else:
                    writer = Writer(fps=fps, metadata=dict(artist='simulate.py'))
                    ani.save(str(outp), writer=writer)
        except Exception as e:
            print('Save error:', e)

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
