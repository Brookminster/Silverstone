import polars as pl
import numpy as np
import os
from pathlib import Path

import Utils.config as configs
import Utils.grid_sampling as grid_sam

class Grid:
    def __init__(
        self,
        config:configs.config,
        csv_path:str):
        self.csv_path = csv_path
        self.gc = config.gc
        self.tc = config.tc
        #self.fc = configs.folder_config

    def track_data_import(self):
        df = pl.read_csv(self.csv_path)
        #The length of Silverstone the track is 5889m
        return df 
        
    def lane_creation(self,ldf:pl.LazyFrame):
        N_LANES = self.gc.N_LANES
        lane_factors = [-0.5 + i / (N_LANES - 1) for i in range(N_LANES)] if N_LANES > 1 else [ 0.0 ]
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
        ]).with_columns(
                [(pl.col("x_m") - pl.lit(f) * pl.col("width") * pl.col("y_udir")).alias(f"x_lane{i}") for i, f in enumerate(lane_factors)] +
                [(pl.col("y_m") + pl.lit(f) * pl.col("width") * pl.col("x_udir")).alias(f"y_lane{i}") for i, f in enumerate(lane_factors)]
            )

        #aldf = aldf.with_columns(Grid.get_curvature("x_m", "y_m"))
        return aldf

    def track_thinning(self,lanes_ldf:pl.LazyFrame):
        lanes_df = lanes_ldf.collect()
        initial_length = len(lanes_df)
        #lanes_df = lanes_df[::2]
        lanes_df = lanes_df.filter(pl.int_range(0, pl.len()) % 2 == 0)
        
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

    def save_frame(self,df:pl.LazyFrame,name):
        #path = Path(FOLDER + f"\\TrackLanes_{N_LANES}.parquet")
        path = Path(configs.folder_config.DATA_FOLDER +"/" + name + f"_l{self.gc.N_LANES}.parquet")
        df.write_parquet(path)
        return path 

    #distance between lanes in subsequent time steps
    def make_lane_distance_columns(l0,l1):
        x_col0 = pl.col("x_lane"+str(l0))
        x_col1 = pl.col("x_lane"+str(l1)).shift(-1)
        y_col0 = pl.col("y_lane"+str(l0))
        y_col1 = pl.col("y_lane"+str(l1)).shift(-1)
        dist = ((x_col1-x_col0).pow(2) + (y_col1-y_col0).pow(2)).sqrt()
        return dist.alias(f"d_l{l0}_l{l1}")
    
    #time cost
    def make_time_costs(gc,l0,l1,s0,s1):
        av_speed = 0.5*(gc.index_to_speed(s0) + gc.index_to_speed(s1))
        d01_col = pl.col(f"d_l{l0}_l{l1}")
        tc = d01_col/pl.lit(av_speed) 
        return tc.alias(f"tc_l{l0}s{s0}_l{l1}s{s1}")
    
    #acceleration
    def make_acceleration(gc,l0,s0,l1,s1):
        speed_delta = gc.index_to_speed(s1) - gc.index_to_speed(s0)
        time_cost = pl.col(f"tc_l{l0}s{s0}_l{l1}s{s1}")
        eps=1e-12
        acc = pl.when(time_cost < eps).then(0.0).otherwise(pl.lit(speed_delta)/time_cost)
        return acc.alias(f"acc_l{l0}s{s0}_l{l1}s{s1}")
    
    #angular velocity
    def make_kappa(lp,l0,d0,l1,d1):
        ln = l1
        x0 = pl.col("x_lane"+str(lp)).shift(1)
        x1 = pl.col("x_lane"+str(l0))
        x2 = pl.col("x_lane"+str(ln)).shift(-1)
        y0 = pl.col("y_lane"+str(lp)).shift(1)
        y1 = pl.col("y_lane"+str(l0))
        y2 = pl.col("y_lane"+str(ln)).shift(-1)
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
    
    def augment_frame_for_optimisation(grid,lanes_ldf:pl.LazyFrame):
        lane_distance_columns = [Grid.make_lane_distance_columns(l0,l1) for (l0,l1) in grid.tc.distance_pairs]
        # lane_distance_columns_names = [f"d_l{l0}_l{l1}" for (l0,l1) in pc.distance_pairs]
        time_costs_columns = [Grid.make_time_costs(grid.gc,l0,l1,s0,s1) for (l0,s0,l1,s1) in grid.tc.lanes_speed_range]
        # time_costs_columns_names = [f"tc_l{l0}s{s0}_l{l1}s{s1}" for (l0,s0,l1,s1) in pc.lanes_speed_range]
        acceleration_columns = [Grid.make_acceleration(grid.gc,l0,s0,l1,s1) for (l0,s0,l1,s1) in grid.tc.lanes_speed_range]
        # acceleration_columns_names = [f"acc_l{l0}s{s0}_l{l1}s{s1}" for (l0,s0,l1,s1) in pc.lanes_speed_range]
        kappa_columns = [Grid.make_kappa(grid.tc.get_previous_lane(l0,d0),l0,d0,l1,d1) for (l0,d0,l1,d1) in grid.tc.lanes_dir_range]
        # kappa_columns_names = [f"ka_l{l0}d{d0}_l{l1}d{d1}" for (l0,d0,l1,d1) in lanes_dir_range]
        opt_ldf = lanes_ldf.with_columns(
            lane_distance_columns).with_columns(
                time_costs_columns).with_columns(acceleration_columns + kappa_columns).fill_null(0.0)

        return opt_ldf 
    
    def data_generation(self, config, name, save=False,verbose=False):
        if verbose: print("***** Importing Track Data *****")
        track_df = self.track_data_import()
        if verbose: print("***** Optimal sampling *****")
        reduced_track = grid_sam.Grid_Sampling.frame_thining(config,track_df).lazy()
        if verbose: print("***** Creating Lanes *****")
        track_lanes_ldf = self.lane_creation(reduced_track)

        # path = Path(FOLDER + r"\\TrackPlot.html")
        # Utils.visualisation.make_just_track_plot(track_lanes_ldf.collect(),path)
        # if verbose: print("***** Thinging track *****")
        # filtered_track_lanes_df = self.track_thinning(track_lanes_ldf)
        #filtered_track_lanes_df = track_lanes_ldf
        filtered_track_lanes_df = track_lanes_ldf
        if verbose: print("***** columns for optmisation *****")
        opt_ldf = self.augment_frame_for_optimisation(filtered_track_lanes_df.lazy())
        if verbose: print("***** Collecting opt data *****")
        opt_df = opt_ldf.collect() 
        if verbose: print("***** Writing opt_df *****")
        opt_data_path = None
        if save:
            opt_data_path = self.save_frame(opt_df,name)
        return opt_df, opt_data_path
