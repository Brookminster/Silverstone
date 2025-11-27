import polars as pl
import numpy as np
import os
from pathlib import Path

import Utils.constants
from Utils.constants import FOLDER, DATA_FOLDER
from Utils.constants import N_LANES, N_SPEEDS, N_TIME_LIMIT, N_DIRECTIONS, N_SPEED_DEVIATION, N_LANE_DEVIATION, N_TIME_LIMIT
from Utils.constants import MAX_COST_VALUE
from Utils.constants import index_to_speed
from Utils.state_navigation import l0_range, l1_range, d0_range, d1_range, s0_range, s1_range, index_to_speed
from Utils.state_navigation import action_range, distance_pairs, get_next_lane, get_previous_lane, lanes_speed_range, speeds_range,lanes_dir_range
from Utils.state_navigation import get_index_from_policy_tuple, get_policy_tuple_from_index
print(f"N_LANES: {N_LANES} N_SPEEDS:{N_SPEEDS} N_DIRECTIONS:{N_DIRECTIONS}")
print(f"State space {N_LANES*N_SPEEDS*N_DIRECTIONS} N_SPEED_DEVIATION:{N_SPEED_DEVIATION} N_LANE_DEVIATION:{N_LANE_DEVIATION}")

def track_data_import():
    path = Path(FOLDER + r"\\TrackData.csv")
    df = pl.read_csv(path)
    #The length of the track is 5889m
    return df.head(N_TIME_LIMIT) 

def lane_creation(ldf:pl.LazyFrame):
    print("***** Importing Track Data *****")
    #Create and add the lanes
    aldf = ldf.with_columns([
        (pl.col("x_m").shift(-1).fill_null(strategy="forward") - pl.col("x_m").shift(1).fill_null(strategy="backward")).alias("x_dir"),
        (pl.col("y_m").shift(-1).fill_null(strategy="forward") - pl.col("y_m").shift(1).fill_null(strategy="backward")).alias("y_dir"),
        (pl.lit(1.2)*((pl.col("w_tr_left_m") + pl.col("w_tr_right_m")))).alias("width")
    ]).with_columns([
        (pl.col("x_dir")*pl.col("x_dir") + pl.col("y_dir")*pl.col("y_dir")).sqrt().alias("norm_dir")
    ]).with_columns([
        (pl.col("x_dir")/pl.col("norm_dir")).alias("x_udir"),
        (pl.col("y_dir")/pl.col("norm_dir")).alias("y_udir")
    ])

    lane_factors = [-0.5 + i / (N_LANES - 1) for i in range(N_LANES)] if N_LANES > 1 else [ 0.0 ]
    #print(lane_factors)
    aldf = aldf.with_columns(
            [(pl.col("x_m") - pl.lit(f) * pl.col("width") * pl.col("y_udir")).alias(f"x_lane{i}") for i, f in enumerate(lane_factors)] +
            [(pl.col("y_m") + pl.lit(f) * pl.col("width") * pl.col("x_udir")).alias(f"y_lane{i}") for i, f in enumerate(lane_factors)]
        )
    def get_curvature():
        x_m0 = pl.col("x_m").shift(-1).fill_null(strategy="forward")
        x_m1 = pl.col("x_m")
        x_m2 = pl.col("x_m").shift(1).fill_null(strategy="backward")
        y_m0 = pl.col("y_m").shift(-1).fill_null(strategy="forward")
        y_m1 = pl.col("y_m")
        y_m2 = pl.col("y_m").shift(1).fill_null(strategy="backward")
        d0 = (x_m1 - x_m0).pow(2) + (y_m1 - y_m0).pow(2)
        d1 = (x_m2 - x_m1).pow(2) + (y_m2 - y_m1).pow(2)
        snd_diff = (x_m0 - 2*x_m1 + x_m2).pow(2) + (y_m0 - 2*y_m1 + y_m2).pow(2)
        prod = d0*d1
        return pl.when(prod != 0.0).then((snd_diff/prod).sqrt()).otherwise(pl.lit(0.0)).alias(f"kappa_mid")
    aldf = aldf.with_columns(get_curvature())
    return aldf

def track_thinning(lanes_ldf:pl.LazyFrame):
    lanes_df = lanes_ldf.collect()
    initial_length = len(lanes_df)
    #lanes_df = lanes_df[::2]
    filtered_lanes_df = (
        lanes_df
        .with_row_index("idx")  # adds 0,1,2,... as 'idx'
        .filter(
            (pl.col("kappa_mid") >= 0.005) | 
            ((pl.col("kappa_mid") >= 0.0008) & (pl.col("idx") % 3) != 0) |
            ((pl.col("idx") % 3) == 0)
        )
        .drop("idx")  # optional: clean up
    )
    filtered_length = len(filtered_lanes_df)
    print(f"Went from length {initial_length} to length {filtered_length}")
    return filtered_lanes_df

def save_frame(df:pl.LazyFrame,name):
    #path = Path(FOLDER + f"\\TrackLanes_{N_LANES}.parquet")
    path = Path(FOLDER +"/" + name + f"_l{N_LANES}.parquet")
    df.write_parquet(path)
    return path 



#distance between lanes in subsequent time steps
def make_lane_distance_columns(l0,l1):
    x_col0 = pl.col("x_lane"+str(l0))
    x_col1 = pl.col("x_lane"+str(l1)).shift(-1).fill_null(strategy="forward")
    y_col0 = pl.col("y_lane"+str(l0))
    y_col1 = pl.col("y_lane"+str(l1)).shift(-1).fill_null(strategy="forward")
    dist = ((x_col1-x_col0).pow(2) + (y_col1-y_col0).pow(2)).sqrt()
    return dist.alias(f"d_l{l0}_l{l1}")
lane_distance_columns = [make_lane_distance_columns(l0,l1) for (l0,l1) in distance_pairs]
lane_distance_columns_names = [f"d_l{l0}_l{l1}" for (l0,l1) in distance_pairs]

# #average speed
# def make_average_speed(s0,s1):
#     avspeed = 0.5*(index_to_speed(s0) + index_to_speed(s1))
#     avs = pl.lit(avspeed)
#     return avs.alias(f"as_s{s0}_s{s1}")
# average_speeds_columns = [make_average_speed(s0,s1) for s0 in s0_range() for s1 in s1_range(s0)]
# average_speeds_columns_names = [f"as_s{s0}_s{s1}" for (s0,s1) in speeds_range]

#time cost
def make_time_costs(l0,l1,s0,s1):
    av_speed = 0.5*(index_to_speed(s0) + index_to_speed(s1))
    d01_col = pl.col(f"d_l{l0}_l{l1}")
    ratio = pl.when(av_speed > 0).then(d01_col/av_speed).otherwise(MAX_COST_VALUE)
    eps=1e-12
    tc = pl.when(d01_col < eps).then(pl.lit(0.0)).otherwise(ratio)
    return tc.alias(f"tc_l{l0}s{s0}_l{l1}s{s1}")
time_costs_columns = [make_time_costs(l0,l1,s0,s1) for (l0,s0,l1,s1) in lanes_speed_range]
time_costs_columns_names = [f"tc_l{l0}s{s0}_l{l1}s{s1}" for (l0,s0,l1,s1) in lanes_speed_range]

#acceleration
def make_acceleration(l0,s0,l1,s1):
    speed_delta = index_to_speed(s1) - index_to_speed(s0)
    time_cost = pl.col(f"tc_l{l0}s{s0}_l{l1}s{s1}")
    ratio = pl.when(time_cost > 0).then(pl.lit(speed_delta)/time_cost).otherwise(MAX_COST_VALUE)
    eps=1e-12
    acc = pl.when(speed_delta < eps).then(pl.lit(0.0)).otherwise(ratio)
    return acc.alias(f"acc_l{l0}s{s0}_l{l1}s{s1}")
acceleration_columns = [make_acceleration(l0,s0,l1,s1) for (l0,s0,l1,s1) in lanes_speed_range]
acceleration_columns_names = [f"acc_l{l0}s{s0}_l{l1}s{s1}" for (l0,s0,l1,s1) in lanes_speed_range]

#inner product to get a sense of curvature
def make_ip(l0,d0,l1,d1):
    ln = l1
    lp = get_previous_lane(l0,d0)
    x_m0 = pl.col("x_lane"+str(lp)).shift(-1).fill_null(strategy="forward")
    x_m1 = pl.col("x_lane"+str(l0))
    x_m2 = pl.col("x_lane"+str(ln)).shift(1).fill_null(strategy="backward")
    y_m0 = pl.col("y_lane"+str(lp)).shift(-1).fill_null(strategy="forward")
    y_m1 = pl.col("y_lane"+str(l1))
    y_m2 = pl.col("y_lane"+str(ln)).shift(1).fill_null(strategy="backward")
    iprod = (x_m1 - x_m0)*(x_m2 - x_m1) + (y_m1 - y_m0)*(y_m2 - y_m1)
    di0 = (x_m1 - x_m0).pow(2) + (y_m1 - y_m0).pow(2)
    di1 = (x_m2 - x_m1).pow(2) + (y_m2 - y_m1).pow(2)
    prod = di0*di1
    kappa = pl.when((iprod != 0.0) & (x_m0 != x_m1) & (x_m1 != x_m2)).then((iprod/(prod.sqrt()))).otherwise(pl.lit(1.0))
    av = kappa
    return av.alias(f"ip_l{l0}d{d0}_l{l1}d{d1}")
ip_columns = [make_ip(l0,d0,l1,d1) for (l0,d0,l1,d1) in lanes_dir_range]
ip_columns_names = [f"ip_l{l0}d{d0}_l{l1}d{d1}" for (l0,d0,l1,d1) in lanes_dir_range]

#angular velocity
def make_kappa(l0,d0,l1,d1):
    ln = l1
    lp = get_previous_lane(l0,d0)
    x0 = pl.col("x_lane"+str(lp)).shift(-1).fill_null(strategy="forward")
    x1 = pl.col("x_lane"+str(l0))
    x2 = pl.col("x_lane"+str(ln)).shift(1).fill_null(strategy="backward")
    y0 = pl.col("y_lane"+str(lp)).shift(-1).fill_null(strategy="forward")
    y1 = pl.col("y_lane"+str(l1))
    y2 = pl.col("y_lane"+str(ln)).shift(1).fill_null(strategy="backward")
    eps=1e-12
    a = ((x1 - x2)**2 + (y1 - y2)**2).sqrt()
    b = ((x0 - x2)**2 + (y0 - y2)**2).sqrt()
    c = ((x0 - x1)**2 + (y0 - y1)**2).sqrt()
    # signed twice-area via cross product
    area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    A = area2.abs() * 0.5
    den = a * b * c
    num = 4.0 * A
    kappa = pl.when(den > eps).then(num / den).otherwise(pl.lit(0.0))       # or 0.0 if you prefer
    return kappa.alias(f"ka_l{l0}d{d0}_l{l1}d{d1}")
kappa_columns = [make_kappa(l0,d0,l1,d1) for (l0,d0,l1,d1) in lanes_dir_range]
kappa_columns_names = [f"ka_l{l0}d{d0}_l{l1}d{d1}" for (l0,d0,l1,d1) in lanes_dir_range]

def augment_frame_for_optimisation(lanes_ldf:pl.LazyFrame):
    opt_ldf = lanes_ldf.with_columns(
        lane_distance_columns).with_columns(
             time_costs_columns).with_columns(acceleration_columns + kappa_columns)

    return opt_ldf 

def np_action_range_frames_for_name(df:pl.LazyFrame, name, save=True):
    N_TIME = len(df)
    arr = np.full((N_LANES*N_SPEEDS*N_DIRECTIONS,N_LANES*N_SPEEDS*N_DIRECTIONS,N_TIME), MAX_COST_VALUE)
    def state_cost_tuple_to_index(l0,s0,d0):
        index = l0*N_SPEEDS*N_DIRECTIONS + s0*N_DIRECTIONS + d0 
        return index
    def name_fun(l0,s0,d0,l1,s1,d1):
        return f"{name}_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}"
    for (l0,s0,d0,l1,s1,d1) in action_range:
        index0 = state_cost_tuple_to_index(l0,s0,d0) 
        index1 = state_cost_tuple_to_index(l1,s1,d1)
        arr[index0][index1] = df[name_fun(l0,s0,d0,l1,s1,d1)].to_numpy()
    arr = arr.transpose(2,0,1)
    path = None
    if save:
        path = Path(DATA_FOLDER + f"\\{name}_{N_LANES}")
        np.save(path, arr)
    return arr, path

def np_lanes_dir_range_frames_for_name(df:pl.LazyFrame, name,save=True):
    N_TIME = len(df)
    arr = np.full((N_LANES*N_DIRECTIONS,N_LANES*N_DIRECTIONS,N_TIME), MAX_COST_VALUE)
    def state_cost_tuple_to_index(l0,d0):
        index = l0*N_DIRECTIONS + d0 
        return index
    def name_fun(l0,d0,l1,d1):
        return f"{name}_l{l0}d{d0}_l{l1}d{d1}"
    for (l0,d0,l1,d1) in lanes_dir_range:
        index0 = state_cost_tuple_to_index(l0,d0) 
        index1 = state_cost_tuple_to_index(l1,d1)
        arr[index0][index1] = df[name_fun(l0,d0,l1,d1)].to_numpy()
    arr = arr.transpose(2,0,1)
    path = None
    if save:
        path = Path(DATA_FOLDER + f"\\{name}_{N_LANES}")
        np.save(path, arr)
    return arr, path

def np_lanes_speed_range_frames_for_name(df:pl.LazyFrame, name,save=True):
    N_TIME = len(df)
    arr = np.full((N_LANES*N_SPEEDS,N_LANES*N_SPEEDS,N_TIME), MAX_COST_VALUE)
    def state_cost_tuple_to_index(l0,s0):
        index = l0*N_SPEEDS + s0 
        return index
    def name_fun(l0,s0,l1,s1):
        return f"{name}_l{l0}s{s0}_l{l1}s{s1}"
    for (l0,s0,l1,s1) in lanes_speed_range:
        index0 = state_cost_tuple_to_index(l0,s0) 
        index1 = state_cost_tuple_to_index(l1,s1)
        arr[index0][index1] = df[name_fun(l0,s0,l1,s1)].to_numpy()
    arr = arr.transpose(2,0,1)
    path = None
    if save:
        path = Path(DATA_FOLDER + f"\\{name}_{N_LANES}")
        np.save(path, arr)
    return arr, path

def np_distance_pairs_frames_for_name(df:pl.LazyFrame, name, save=True):
    N_TIME = len(df)
    arr = np.full((N_LANES,N_LANES,N_TIME), MAX_COST_VALUE)
    def name_fun(l0,l1):
        return f"{name}_l{l0}_l{l1}"
    for (l0,l1) in distance_pairs:
        index0 = l0 
        index1 = l1
        arr[index0][index1] = df[name_fun(l0,l1)].to_numpy()
    arr = arr.transpose(2,0,1)
    path = None
    if save:
        path = Path(DATA_FOLDER + f"\\{name}_{N_LANES}")
        np.save(path, arr)
    return arr, path

def np_speeds_range_frames_for_name(df:pl.LazyFrame, name, save=True):
    N_TIME = len(df)
    arr = np.full((N_LANES,N_LANES,N_TIME), MAX_COST_VALUE)
    def name_fun(l0,l1):
        return f"{name}_l{l0}_l{l1}"
    for (l0,l1) in action_range:
        index0 = l0 
        index1 = l1
        arr[index0][index1] = df[name_fun(l0,l1)].to_numpy()
    arr = arr.transpose(2,0,1)
    path = None
    if save:
        path = Path(DATA_FOLDER + f"\\{name}_{N_LANES}")
        np.save(path, arr)
    return arr, path


# def make_numpy_frames_for_optimiser(opt_df:pl.DataFrame):
#     #Tools to navigate the cost array
#     N_TIME = len(opt_df)
#     def make_cost_array(): return np.full((N_LANES*N_SPEEDS*N_DIRECTIONS,N_LANES*N_SPEEDS*N_DIRECTIONS,N_TIME), MAX_COST_VALUE)
#     tc_arr = make_cost_array()
#     acc_arr = make_cost_array()
#     as_arr = make_cost_array()

#     def state_cost_tuple_to_index(l0,s0,d0):
#         index = l0*N_SPEEDS*N_DIRECTIONS + s0*N_DIRECTIONS + d0 
#         return index

#     for (l0,s0,d0,l1,s1,d1) in action_range:
#         index0 = state_cost_tuple_to_index(l0,s0,d0) 
#         index1 = state_cost_tuple_to_index(l1,s1,d1)
#         tc_arr[index0][index1] = opt_df[f"tc_l{l0}s{s0}_l{l1}s{s1}"].to_numpy()
#         acc_arr[index0][index1] = opt_df[f"acc_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}"].to_numpy()
#         as_arr[index0][index1] = opt_df[f"as_s{s0}_s{s1}"].to_numpy()
    
#     #makes time the first index
#     tc_arr = tc_arr.transpose(2,0,1)
#     as_arr = as_arr.transpose(2,0,1)
#     acc_arr = acc_arr.transpose(2,0,1)
    
#     tc_path = Path(FOLDER + f"\\tc_{N_LANES}") 
#     as_path = Path(FOLDER + f"\\as_{N_LANES}") 
#     acc_path = Path(FOLDER + f"\\acc_{N_LANES}")
#     np.save(tc_path, tc_arr)
#     np.save(as_path, as_arr)
#     np.save(acc_path, acc_arr)

#     return tc_path, as_path, acc_path

#Help pring the matrix in the optimisaton 
# def print_state(V):
#     for s0 in range(N_SPEEDS):
#         for d0 in range(N_DIRECTIONS):
#             val = [V[get_index_from_policy_tuple(l0,s0,d0)] for l0 in range(N_LANES)]
#             s = " ".join(f"{x:.5e}" for x in np.ravel(val))
#             print(f"s={s0} d={d0}: " + s)

def data_generation(name):
    print("***** Importing Track Data *****")
    track_df = track_data_import()
    track_ldf = track_df.lazy()
    print("***** Creating Lanes *****")
    track_lanes_ldf = lane_creation(track_ldf)

    path = Path(FOLDER + r"\\TrackPlot.html")
    Utils.visualisation.make_just_track_plot(track_lanes_ldf.collect(),path)

    print("***** Thinging track *****")
    filtered_track_lanes_df = track_thinning(track_lanes_ldf)
    #filtered_track_lanes_df = track_lanes_ldf
    print("***** columns for optmisation *****")
    opt_ldf = augment_frame_for_optimisation(filtered_track_lanes_df.lazy())
    print("***** Collecting opt data *****")
    opt_df = opt_ldf.collect() 
    print("***** Writing opt_df *****")
    opt_data_path = save_frame(opt_df,name)
    print("***** Writing av np-arr*****")
    _,ka_path = np_lanes_dir_range_frames_for_name(opt_df,"ka")
    print("***** Writing acc np-arr*****")
    _,acc_path = np_lanes_speed_range_frames_for_name(opt_df,"acc")
    print("***** Writing tc np-arr*****")
    _,tc_path = np_lanes_speed_range_frames_for_name(opt_df,"tc")
    return opt_data_path,(ka_path,acc_path,tc_path)

def load_np_data(path):
    return np.load(str(path) + ".npy")

def load_pl_data(path):
    return pl.read_parquet(path)

def load_lpl_data(path):
    return pl.scan_parquet(path)
    
# def force_data_generation(opt_df, MASS, tau_h, tau_v, g, MAX_FORCE, mu):
#     f_ldf = add_Forces_and_constraints(opt_df, MASS, tau_h, tau_v, g, MAX_FORCE, mu)
#     print("***** Writing numpy frames costs*****")
#     costs_arr_path = make_cost_numpy_frames_for_optimiser(f_ldf,"cost")
#     return costs_arr_path

# def force_data_generation(opt_df, MASS, tau_h, tau_v, g, MAX_FORCE, mu):
#     f_ldf = add_Forces_and_constraints(opt_df, MASS, tau_h, tau_v, g, MAX_FORCE, mu)
#     print("***** Writing numpy frames costs*****")
#     Fconst_path = make_cost_numpy_frames_for_optimiser(f_ldf,"costs")
#     print("***** Writing numpy frames Facc*****")
#     Facc_path = make_cost_numpy_frames_for_optimiser(f_ldf,"Facc")
#     print("***** Writing numpy frames Fx*****")
#     Fx_path = make_cost_numpy_frames_for_optimiser(f_ldf,"Fx")
#     print("***** Writing numpy frames Fy*****")
#     Fy_path = make_cost_numpy_frames_for_optimiser(f_ldf,"Fy")
#     print("***** Writing numpy frames Fz*****")
#     Fz_path = make_cost_numpy_frames_for_optimiser(f_ldf,"Fz")
#     return (Fconst_path, Facc_path, Fx_path, Fy_path, Fz_path)