import polars as pl
import numpy as np
import os
from pathlib import Path

import Utils.constants
from Utils.constants import FOLDER, DATA_FOLDER
from Utils.constants import N_LANES, N_SPEEDS, N_TIME_LIMIT, N_DIRECTIONS, N_SPEED_DEVIATION, N_LANE_DEVIATION, N_TIME_LIMIT
from Utils.constants import MAX_COST_VALUE
from Utils.state_navigation import l0_range, l1_range, d0_range, d1_range, s0_range, s1_range, index_to_speed
from Utils.state_navigation import action_range, distance_pairs, get_next_lane, get_previous_lane, lanes_speed_range,lanes_dir_range
from Utils.state_navigation import get_index_from_policy_tuple, get_policy_tuple_from_index, get_index_from_ls_double,get_index_from_ld_double
import Utils.visualisation as vis

def add_Force_requiremments( M, g, tau_h, tau_v, mu, F_MAX_ACC):
    s_max = index_to_speed(N_SPEEDS-1)
    s_max_delta =  index_to_speed(N_SPEEDS-1) -  index_to_speed(N_SPEEDS-2) 
    max_acc = s_max_delta*s_max_delta/5.0
    max_speed = index_to_speed(max(s0_range()))
    force = tau_h*max_speed*max_speed + M*max_acc
    print("Force required for max speed", force)


#const_columns_names = [f"const_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}" for (l0,s0,d0,l1,s1,d1) in action_range]
def add_Forces_and_constraints(opt_ldf:pl.LazyFrame, M, g, tau_h, tau_v, mu, F_MAX_ACC, F_X_NEG_SCALE, F_X_MAX_SCALE, F_Y_MAX_SCALE):
    print(f"M:{M} M:{g} tau_h:{tau_h} tau_v:{tau_v} mu:{mu} F_MAX_ACC:{F_MAX_ACC} F_MAX_ACC:{F_X_NEG_SCALE} F_MAX_ACC:{F_X_MAX_SCALE} F_MAX_ACC:{F_Y_MAX_SCALE} ")

    all_cols: list[pl.Expr] = []
    for (l0,s0,d0,l1,s1,d1) in action_range:
        #data
        a = pl.col(f"acc_l{l0}s{s0}_l{l1}s{s1}")
        kappa = pl.col(f"ka_l{l0}d{d0}_l{l1}d{d1}")
        #kappa = pl.col(f"ka_l{l0}d{0}_l{l1}d{0}")
        M_lit = pl.lit(M)
        Mg_lit = pl.lit(M*g)
        tau_h_lit = pl.lit(tau_h)
        tau_v_lit = pl.lit(tau_v)
        F_x_scale_lit = pl.lit(F_X_MAX_SCALE)
        F_x_neg_scale_lit = pl.lit(F_X_NEG_SCALE)
        F_y_scale_lit = pl.lit(F_Y_MAX_SCALE)
        #assembly
        speed = 0.5*(index_to_speed(s0) + index_to_speed(s0)) 
        speed2 = speed*speed
        a_asym = (pl.when(a>0).then(a).otherwise(F_x_neg_scale_lit*a)).abs()
        F_acc = M_lit*a_asym + tau_h_lit*speed2
        F_x = M_lit*abs(a)*F_x_scale_lit 
        F_y = M_lit*kappa*speed2*F_y_scale_lit
        F_z = pl.lit(mu)*(Mg_lit + tau_v_lit*speed2)
        F_t = (F_x*F_x + F_y*F_y) - (F_z*F_z)
        #make cost
        tc = pl.col(f"tc_l{l0}s{s0}_l{l1}s{s1}")
        cost = pl.when((F_acc < F_MAX_ACC) & (F_t < 0.0)).then(tc).otherwise(pl.lit(MAX_COST_VALUE))
        cost_col = cost.alias(f"cost_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}")
        Facc_col = F_acc.alias(f"Facc_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}")
        Fx_col   = F_x.alias(f"Fx_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}")
        Fy_col   = F_y.alias(f"Fy_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}")
        Fz_col   = F_z.alias(f"Fz_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}")
        #cost = pl.when((F_acc < F_MAX_ACC) & ((F_x*F_x + F_y*F_y) < F_MAX_T*F_MAX_T)).then(tc).otherwise(pl.lit(MAX_COST_VALUE))
        #cost = pl.when(F_acc < F_MAX_ACC).then(tc).otherwise(pl.lit(MAX_COST_VALUE))
        all_cols.extend([Facc_col, Fx_col, Fy_col, Fz_col, cost_col])
    #cost_columns = [make_costs(l0,s0,d0,l1,s1,d1) for (l0,s0,d0,l1,s1,d1) in action_range]
    
    f_ldf = opt_ldf.with_columns(all_cols)
    #f_ldf = opt_ldf.with_columns(Facc_columns + Fx_columns + Fy_columns + Fz_columns + cost_columns)
    return f_ldf


def backward_optimiser(costs_arr):
    costs = costs_arr
    gamma = 1.0
    T, S, A = costs.shape
    #print("Sizes: ",T,S,A)
    V = np.zeros((T, S))
    policy = np.zeros((T, S), dtype=int)
    next_state = np.tile(np.arange(S), (S, 1))

    # terminal time
    V[-1] = np.min(costs[-1], axis=1)        # min over actions
    policy[-1] = np.argmin(costs[-1], axis=1)
    #print("Start optimisation at time stamp", T-1)

    # backward induction
    for t in range(T - 2, -1, -1):
        #if t%10==0: print("Time stamp", t)
        # q[t, s, a] = C[t, s, a] + gamma * V[t+1, s]
        # We need to broadcast V[t+1] to match actions axis
        #cont = gamma * V[t + 1][:, None]     # shape (S, 1) # if we can not change state
        cont = gamma * V[t + 1][next_state]
        q = costs[t] + cont                  # shape (S, A)
        policy[t] = np.argmin(q, axis=1)
        V[t] = np.min(q, axis=1)
    return policy,V

def uncertainty_matrix(theta):
    N = N_LANES*N_SPEEDS*N_DIRECTIONS
    A = np.full((N,N), 0.0)
    for index in range(N):
        A[index,index] = 1.0  
    for l0 in l0_range():
        for s0 in s0_range():
            for d0 in d0_range(l0):
                j = get_index_from_policy_tuple(l0,s0,d0)
                a0 = a1 = a2 = a3 = a4 = 0.0
                if j-2 >= 0 and j-2 < N: a0 = theta*theta/16.0
                if j-1 >= 0 and j-1 < N: a1 = theta*theta/4.0
                if j >= 0 and j < N:     a2 = 1.0-5.0*theta*theta/8.0
                if j+1 >= 0 and j+1 < N: a3 = theta*theta/4.0
                if j+2 >= 0 and j+2 < N: a4 = theta*theta/16.0
                at = a0 + a1 + a2 + a3 + a4
                ##Fill out weights
                if j-2 >= 0 and j-2 < N:    A[j-2,j] = a0/at  
                if j-1 >= 0 and j-1 < N:    A[j-1,j] = a1/at
                if j >= 0 and j < N:        A[j,j] = a2/at
                if j+1 >= 0 and j+1 < N:    A[j+1,j] = a3/at  
                if j+2 >= 0 and j+2 < N:    A[j+2,j] = a4/at  
    return A

def backward_optimiser_with_uncertainty(costs_arr,theta):
    costs = costs_arr
    gamma = 1.0
    T, S, A = costs.shape
    #print("Sizes: ",T,S,A)
    V = np.zeros((T, S))
    policy = np.zeros((T, S), dtype=int)
    next_state = np.tile(np.arange(S), (S, 1))

    # uncertainty matrix: P(executed=j | intended=i)
    A = uncertainty_matrix(theta)
    # terminal time
    V[-1] = np.min(costs[-1], axis=1)        # min over actions
    policy[-1] = np.argmin(costs[-1], axis=1)
    print("Start optimisation at time stamp", T-1)

    # backward induction
    for t in range(T - 2, -1, -1):
        #if t%10==0: print("Time stamp", t)
        # q[t, s, a] = C[t, s, a] + gamma * V[t+1, s]
        # We need to broadcast V[t+1] to match actions axis
        #cont = gamma * V[t + 1][:, None]     # shape (S, 1) # if we can not change state
        cont = gamma * V[t + 1][next_state]
        base = costs[t] + cont # shape (S, A)
        q = base@A
        policy[t] = np.argmin(q, axis=1)
        V[t] = np.min(q, axis=1)
    return policy,V

def extract_optimal_trajectory(policy,V):
    N_TIME = len(policy)
    l0 = int(N_LANES/2)
    s0 = 0
    d0 = 0
    optmal_path = np.full((N_TIME,12), 0.0)
    for t in range(N_TIME):
        p0 = get_index_from_policy_tuple(l0,s0,d0)
        #print(index)
        cost = V[t][p0]
        p1 = policy[t][p0]
        l1,s1,d1 = get_policy_tuple_from_index(p1)
        optmal_path[t,0] = int(t)
        optmal_path[t,1] = V[t][p0] - V[t+1][p1] if t + 1 < N_TIME else 0.0
        optmal_path[t,2] = cost
        optmal_path[t,3] = l0
        optmal_path[t,4] = s0
        optmal_path[t,5] = d0
        optmal_path[t,6] = p0
        optmal_path[t,7] = l1
        optmal_path[t,8] = s1
        optmal_path[t,9] = d1
        optmal_path[t,10] = p1
        optmal_path[t,11] = 0.5*(index_to_speed(s0) + index_to_speed(s1))
        
        l0 = l1
        s0 = s1
        d0 = d1
        
    columns = ["index", "cost", "tot_cost", "l0", "s0", "d0", "p0", "l1", "s1", "d1", "p1", "speed"]
    df = pl.from_numpy(optmal_path.transpose(), schema=columns, orient="col")
    
    df = df.with_columns([
        pl.col("index").round(0).cast(pl.Int64),
        pl.col("l0").round(0).cast(pl.Int64),
        pl.col("s0").round(0).cast(pl.Int64),
        pl.col("d0").round(0).cast(pl.Int64),
        pl.col("p0").round(0).cast(pl.Int64),
        pl.col("l1").round(0).cast(pl.Int64),
        pl.col("s1").round(0).cast(pl.Int64),
        pl.col("d1").round(0).cast(pl.Int64),        
        pl.col("p1").round(0).cast(pl.Int64),
    ])
    return df

def add_track_data_to_optimal_trajectory(optimal_trajectory, opt_data):
    aot_df = optimal_trajectory
    #print(f"Test that the frames have the same length {len(aot_df)} {len(opt_data)}")

    N_TIME = len(aot_df)
    #import track data
    x_coord_arr = np.full((N_LANES,N_TIME), MAX_COST_VALUE)
    y_coord_arr = np.full((N_LANES,N_TIME), MAX_COST_VALUE)

    for (l) in range(N_LANES):
        x_coord_arr[l] = opt_data[f"x_lane{l}"].to_numpy()
        y_coord_arr[l] = opt_data[f"y_lane{l}"].to_numpy()
    x_coord_arr = x_coord_arr.transpose()
    y_coord_arr = y_coord_arr.transpose()
    def get_x_coord(t,l):
        return x_coord_arr[int(t),int(l)]
    def get_y_coord(t,l):
        return y_coord_arr[int(t),int(l)]

    aot_df = aot_df.with_columns([
        pl.struct(["index", "l0"]).map_elements(lambda s: get_x_coord(s["index"], s["l0"]),return_dtype=pl.Float64).alias("tr_x"),
        pl.struct(["index", "l0"]).map_elements(lambda s: get_y_coord(s["index"], s["l0"]),return_dtype=pl.Float64).alias("tr_y")
    ]).with_columns([
        (((pl.col("tr_x").shift(-1).fill_null(strategy="forward") - pl.col("tr_x")).pow(2) + 
        (pl.col("tr_y").shift(-1).fill_null(strategy="forward") - pl.col("tr_y")).pow(2)).sqrt()).alias("distance"),
    ]).with_columns([
        (pl.col("distance")/pl.col("cost")).alias("speed_est"),
    ]).with_columns([
        (pl.col("distance")/pl.col("speed")).alias("times_est"),
    ])

    cols_to_add = ["x_lane" + str(index) for index in range(N_LANES)] + ["y_lane" + str(index) for index in range(N_LANES)]
    aot_df = aot_df.hstack(
        opt_data.select(cols_to_add)
    )
    return aot_df
    
def add_Forces_data_to_optimal_trajectory(traj_df, acc, av, M, g, tau_h, tau_v, mu, F_MAX_ACC):
    print(f"M:{M} g:{g} tau_h:{tau_h} tau_v:{tau_v} mu:{mu} F_MAX_ACC:{F_MAX_ACC}")

    def get_acc(t,l0,s0,l1,s1):
        index0 = int(get_index_from_ls_double(l0,s0))
        index1 = int(get_index_from_ls_double(l1,s1))
        return float(acc[int(t),index0,index1])

    def get_ka(t,l0,d0,l1,d1):
        index0 = int(get_index_from_ld_double(l0,d0))
        index1 = int(get_index_from_ld_double(l1,d1))
        return float(av[int(t),index0,index1])

    def F_acc(t,l0,s0,l1,s1):
        a = get_acc(t,l0,s0,l1,s1)
        speed = index_to_speed(s0)
        F = M*(a if a > 0 else -0.5*a) + tau_h*speed*speed
        return F
    
    def F_x(t,l0,s0,l1,s1):
        a = get_acc(t,l0,s0,l1,s1)
        F = M*abs(a)
        return F
    
    def F_y(t,l0,s0,d0,l1,s1,d1):
        ka = get_ka(t,l0,d0,l1,d1)
        speed = index_to_speed(s0)
        F = M*ka*speed*speed
        return F
    
    def F_z(t,s0):
        speed = index_to_speed(s0)
        F = mu*(M*g + tau_v*speed*speed)
        return F

    traj_df = traj_df.with_columns([
        pl.struct(["index", "l0", "s0", "d0", "l1", "s1", "d1"]).map_elements(lambda s: F_acc(s["index"], s["l0"], s["s0"], s["l1"], s["s1"]),return_dtype=pl.Float64).alias("F_acc"),
        pl.struct(["index", "l0", "s0", "d0", "l1", "s1", "d1"]).map_elements(lambda s: F_x(s["index"], s["l0"], s["s0"], s["l1"], s["s1"]),return_dtype=pl.Float64).alias("F_x"),
        pl.struct(["index", "l0", "s0", "d0", "l1", "s1", "d1"]).map_elements(lambda s: F_y(s["index"], s["l0"], s["s0"], s["d0"], s["l1"], s["s1"], s["d1"]),return_dtype=pl.Float64).alias("F_y"),
        pl.struct(["index", "l0", "s0", "d0", "l1", "s1", "d1"]).map_elements(lambda s: F_z(s["index"],s["s0"]),return_dtype=pl.Float64).alias("F_z"),
        pl.lit(F_MAX_ACC).alias("F_MAX_ACC")
    ])
    return traj_df

def make_diagnostic_plots(df, show_plots, run_path):
    print("***** Track plots *****")
    vis.make_track_plot_2(df, show_plots, run_path + r"\\Trajectory.html")

    print("***** Plots with speed and power usage *****")
    x_data = df["index"].to_numpy()
    s_data = df["speed"].to_numpy()
    vis.pl_row_plots(x_data, {"speed":s_data}, "speed", show_plots, run_path + r"\\Speed.html")
    F_acc_data = df["F_acc"].to_numpy()
    F_max_acc_data = df["F_MAX_ACC"].to_numpy()
    vis.pl_row_plots(x_data, {"F_acc":F_acc_data,"MAX_FORCE":F_max_acc_data}, "Acc force Usage", show_plots, run_path + r"\\F_acc.html")
    F_x_data = df["F_x"].to_numpy()
    F_y_data = df["F_y"].to_numpy()
    F_z_data = df["F_z"].to_numpy()
    #F_max_t_data = df["F_MAX_T"].to_numpy()
    vis.pl_row_plots(x_data, {"F_x":F_x_data,"F_y":F_y_data,"F_z":F_z_data}, "Force Usage", show_plots, run_path + r"\\F_usage.html")
    vis.pl_row_plots(x_data, {"F_x*F_x":F_x_data*F_x_data,"F_y*F_y":F_y_data*F_y_data,"F_z*F_z":F_z_data*F_z_data}, "Force Usage", show_plots, run_path + r"\\F2_usage.html")
    return

def make_cost_df(opt_df,MASS, g, tau_h, tau_v, mu, F_MAX_ACC, F_X_NEG_SCALE, F_X_MAX_SCALE, F_Y_MAX_SCALE):
    costs_ldf = add_Forces_and_constraints(opt_df.lazy(), MASS, g, tau_h, tau_v, mu, F_MAX_ACC, F_X_NEG_SCALE, F_X_MAX_SCALE, F_Y_MAX_SCALE)
    costs_df = costs_ldf.collect()
    return costs_df

import book_keeping as bk

def run_opt(costs_df, acc_arr, av_arr, MASS, g, tau_h, tau_v, mu, F_MAX_ACC, F_X_NEG_SCALE, F_X_MAX_SCALE, F_Y_MAX_SCALE, rho):
    print(f"PARAMETERS: M:{MASS} g:{g} tau_h:{tau_h} tau_v:{tau_v} mu:{mu} F_MAX_ACC:{F_MAX_ACC} F_X_NEG_SCALE:{F_X_NEG_SCALE} F_X_MAX_SCALE:{F_X_MAX_SCALE} F_Y_MAX_SCALE:{F_Y_MAX_SCALE} rho:{rho} ")
    print("***** making cost_arr *****")
    costs,_ = bk.np_action_range_frames_for_name(costs_df,"cost",save=False)
    print(f"***** optimal policy with uncertainty {rho} *****")
    if rho == 0.0:
        policy,V = backward_optimiser(costs)
    else:
        policy,V = backward_optimiser_with_uncertainty(costs, rho)
    print("***** extracting optimal trajectory *****")
    optimal_trajectory_df = extract_optimal_trajectory(policy,V)
    print("***** augmenting trajectory with track data *****")
    traj_df = add_track_data_to_optimal_trajectory(optimal_trajectory_df,costs_df)
    print("***** adding force terms to trajectory *****")
    atraj_df = add_Forces_data_to_optimal_trajectory(traj_df, acc_arr, av_arr, MASS, g, tau_h, tau_v, mu, F_MAX_ACC)
    print("*****making plots *****")
    #path_opt.make_diagnostic_plots(atraj_df)
    run_path, setting = bk.init_test_run_folder(FOLDER,N_LANES,N_SPEEDS,N_DIRECTIONS,MASS,g,tau_h,tau_v,mu,F_MAX_ACC,F_X_NEG_SCALE,F_X_MAX_SCALE,F_Y_MAX_SCALE,rho)
    bk.save_frame_to_run_dir(atraj_df, run_path, "trajectory")
    print("***** making plots *****")
    make_diagnostic_plots(atraj_df, False, str(run_path))
    print("***** make result frame *****")
    results_frame = print_opt_result(atraj_df)
    bk.save_frame_to_run_dir(results_frame, run_path, "results")

def print_opt_result(df):
    tot_cost = max(df["tot_cost"])
    tot_cost_robust = sum(list(df["times_est"]))
    max_speed = max(df["s0"])*Utils.constants.INDEX_SPEED_MULTIPLIER
    print(f"tot_cost:{tot_cost} tot_cost_robust={tot_cost_robust} max_speed={max_speed}")
    result_df = pl.DataFrame({
        "tot_cost":tot_cost,
        "tot_cost_robust":tot_cost_robust,
        "max_speed":max_speed
    })
    return result_df


