import Utils.config as configs
import Utils.force_frame as force_frame
import Utils.visualisation as vis
import Utils.book_keeping as book_keeping

import polars as pl
from pathlib import Path
import numpy as np
import os

class Optimisation:

    staticmethod
    def backward_optimiser(costs_arr):
        costs = costs_arr
        gamma = 1.0
        T, S, A = costs.shape
        #print("Sizes: ",T,S,A)
        V = np.zeros((T, S))
        policy = np.zeros((T, S), dtype=int)
        next_state = np.tile(np.arange(S), (S, 1))

        # terminal time
        #V[-1] = np.min(costs[-1], axis=1)        # min over actions
        V[-1] = np.zeros(S)
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

    # At any time we might go -2 to +2 faster
    staticmethod
    def um_even_2_step_speed_error( config:configs.config,
                                    theta:float):
        gc = config.gc
        tc = config.tc
        N = gc.N_LANES*gc.N_SPEEDS*gc.N_DIRECTIONS
        A = np.full((N,N), 0.0)
        for index in range(N):
            A[index,index] = 1.0  
        for l0 in tc.l0_range():
            for s0 in tc.s0_range():
                for d0 in tc.d0_range(l0):
                    j = tc.get_index_from_policy_tuple(l0,s0,d0)
                    a0 = a1 = a2 = a3 = a4 = 0.0
                    if s0-2 >= 0:       a0 = theta*theta/16.0
                    if s0-1 >= 0:       a1 = theta*theta/4.0
                    if s0   >= 0:       a2 = 1.0-5.0*theta*theta/8.0
                    if s0+1 < gc.N_SPEEDS: a3 = theta*theta/4.0
                    if s0+2 < gc.N_SPEEDS: a4 = theta*theta/16.0
                    at = a0 + a1 + a2 + a3 + a4
                    if s0-2 >= 0:
                        i = tc.get_index_from_policy_tuple(l0,s0-2,d0)
                        A[i,j] = a0/at 
                    if s0-1 >= 0:
                        i = tc.get_index_from_policy_tuple(l0,s0-1,d0)
                        A[i,j] = a1/at
                    if s0 >= 0:
                        i = tc.get_index_from_policy_tuple(l0,s0,d0)
                        A[i,j] = a2/at
                    if s0+1 < gc.N_SPEEDS:
                        i = tc.get_index_from_policy_tuple(l0,s0+1,d0)
                        A[i,j] = a3/at
                    if s0+2 < gc.N_SPEEDS:
                        i = tc.get_index_from_policy_tuple(l0,s0+2,d0)
                        A[i,j] = a4/at
        return A

    #This matrix is about casting doubt on breaking. 
    staticmethod
    def um_even_2_step_speed_error( config:configs.config,
                                    theta:float):
        gc = config.gc
        tc = config.tc
        N = gc.N_LANES*gc.N_SPEEDS*gc.N_DIRECTIONS
        A = np.full((N,N), 0.0)
        for index in range(N):
            A[index,index] = 1.0  
        for l0 in gc.l0_range():
            for s0 in gc.s0_range():
                for d0 in gc.d0_range(l0):
                    j = tc.get_index_from_policy_tuple(l0,s0,d0)
                    a0 = a1 = a2 = 0.0
                    if s0-2 >= 0:       a0 = theta*theta/16.0*2.0
                    if s0-1 >= 0:       a1 = theta*theta/4.0*2.0
                    if s0   >= 0:       a2 = 1.0-5.0*theta*theta/8.0*4.0
                    at = a0 + a1 + a2
                    if s0-2 >= 0:
                        i = tc.get_index_from_policy_tuple(l0,s0-2,d0)
                        A[i,j] = a0/at 
                    if s0-1 >= 0:
                        i = tc.get_index_from_policy_tuple(l0,s0-1,d0)
                        A[i,j] = a1/at
                    if s0 >= 0:
                        i = tc.get_index_from_policy_tuple(l0,s0,d0)
                        A[i,j] = a2/at
        return A
    
    #This matrix is about casting doubt on breaking. 
    def um_break_4_step_break_error( config:configs.config,
                                    control_trust:float,
                                    c):
        tc = config.tc
        theta = 1.0 - control_trust
        new_c = c.copy()
        for (l0,s0,d0,l1,s1,d1) in tc.action_range:
            if s1 < s0:
                if s0 - 1 == s1:
                    i = tc.get_index_from_policy_tuple(l0,s0,d0)
                    j = tc.get_index_from_policy_tuple(l1,s1,d1)
                    j1 = tc.get_index_from_policy_tuple(l1,s1+1,d1)
                    a1 = theta*theta/2.0
                    a = 1.0-a1
                    new_c[i,j] = a*c[i,j] + a1*c[i,j1]
                if s0 - 2 == s1:
                    i = tc.get_index_from_policy_tuple(l0,s0,d0)
                    j = tc.get_index_from_policy_tuple(l1,s1,d1)
                    j1 = tc.get_index_from_policy_tuple(l1,s1+1,d1)
                    j2 = tc.get_index_from_policy_tuple(l1,s1+2,d1)
                    a2 = theta*theta/4.0
                    a1 = theta*theta/2.0
                    a = 1.0-a1-a2
                    new_c[i,j] = a*c[i,j] + a1*c[i,j1] + a2*c[i,j2]
                if s0 - 3 == s1:
                    i = tc.get_index_from_policy_tuple(l0,s0,d0)
                    j = tc.get_index_from_policy_tuple(l1,s1,d1)
                    j1 = tc.get_index_from_policy_tuple(l1,s1+1,d1)
                    j2 = tc.get_index_from_policy_tuple(l1,s1+2,d1)
                    j3 = tc.get_index_from_policy_tuple(l1,s1+3,d1)
                    a3 = theta*theta/8.0
                    a2 = theta*theta/4.0
                    a1 = theta*theta/2.0
                    a = 1.0 - a1- a2 - a3
                    new_c[i,j] = a*c[i,j] + a1*c[i,j1] + a2*c[i,j2] + a3*c[i,j3]
                if s0 - 4 == s1:
                    i = tc.get_index_from_policy_tuple(l0,s0,d0)
                    j = tc.get_index_from_policy_tuple(l1,s1,d1)
                    j1 = tc.get_index_from_policy_tuple(l1,s1+1,d1)
                    j2 = tc.get_index_from_policy_tuple(l1,s1+2,d1)
                    j3 = tc.get_index_from_policy_tuple(l1,s1+3,d1)
                    j4 = tc.get_index_from_policy_tuple(l1,s1+4,d1)
                    a4 = theta*theta/16.0
                    a3 = theta*theta/8.0
                    a2 = theta*theta/4.0
                    a1 = theta*theta/2.0
                    a = 1.0 - a1 - a2 - a3 - a4
                    new_c[i,j] = a*c[i,j] + a1*c[i,j1] + a2*c[i,j2] + a3*c[i,j3] + a4*c[i,j4]
                if s0 + 1 == s1:
                    i = tc.get_index_from_policy_tuple(l0,s0,d0)
                    j = tc.get_index_from_policy_tuple(l1,s1,d1)
                    j1 = tc.get_index_from_policy_tuple(l1,s1-1,d1)
                    a1 = theta*theta/2.0
                    a = 1.0-a1
                    new_c[i,j] = a*c[i,j] + a1*c[i,j1]
                if s0 + 2 == s1:
                    i = tc.get_index_from_policy_tuple(l0,s0,d0)
                    j = tc.get_index_from_policy_tuple(l1,s1,d1)
                    j1 = tc.get_index_from_policy_tuple(l1,s1-1,d1)
                    j2 = tc.get_index_from_policy_tuple(l1,s1-2,d1)
                    a2 = theta*theta/4.0
                    a1 = theta*theta/2.0
                    a = 1.0-a1-a2
                    new_c[i,j] = a*c[i,j] + a1*c[i,j1] + a2*c[i,j2]
                if s0 + 3 == s1:
                    i = tc.get_index_from_policy_tuple(l0,s0,d0)
                    j = tc.get_index_from_policy_tuple(l1,s1,d1)
                    j1 = tc.get_index_from_policy_tuple(l1,s1-1,d1)
                    j2 = tc.get_index_from_policy_tuple(l1,s1-2,d1)
                    j3 = tc.get_index_from_policy_tuple(l1,s1-3,d1)
                    a3 = theta*theta/8.0
                    a2 = theta*theta/4.0
                    a1 = theta*theta/2.0
                    a = 1.0 - a1- a2 - a3
                    new_c[i,j] = a*c[i,j] + a1*c[i,j1] + a2*c[i,j2] + a3*c[i,j3]
                if s0 + 4 == s1:
                    i = tc.get_index_from_policy_tuple(l0,s0,d0)
                    j = tc.get_index_from_policy_tuple(l1,s1,d1)
                    j1 = tc.get_index_from_policy_tuple(l1,s1-1,d1)
                    j2 = tc.get_index_from_policy_tuple(l1,s1-2,d1)
                    j3 = tc.get_index_from_policy_tuple(l1,s1-3,d1)
                    j4 = tc.get_index_from_policy_tuple(l1,s1-4,d1)
                    a4 = theta*theta/16.0
                    a3 = theta*theta/8.0
                    a2 = theta*theta/4.0
                    a1 = theta*theta/2.0
                    a = 1.0 - a1 - a2 - a3 - a4
                    new_c[i,j] = a*c[i,j] + a1*c[i,j1] + a2*c[i,j2] + a3*c[i,j3] + a4*c[i,j4]
                    
                    
        return new_c

    staticmethod
    def backward_optimiser_with_uncertainty(config:configs.config,
                                            costs_arr):
        costs = costs_arr
        gamma = 1.0
        T, S, A = costs.shape
        V = np.zeros((T, S))
        policy = np.zeros((T, S), dtype=int)
        next_state = np.tile(np.arange(S), (S, 1)) #square matrix where each row is [0,..,S-1].
        V[-1] = np.zeros(S) #np.min(costs[-1], axis=1)
        policy[-1] = np.argmin(costs[-1], axis=1)
        #print("Start optimisation at time stamp", T-1)
        # backward induction
        for t in range(T - 2, -1, -1):
            #if t%10==0: print("Time stamp", t)
            # q[t, s, a] = C[t, s, a] + gamma * V[t+1, s]
            # We need to broadcast V[t+1] to match actions axis
            #cont = gamma * V[t + 1][:, None]     # shape (S, 1) # if we can not change state
            #S×S array where each row is equal to V[t + 1].
            cont = gamma * V[t + 1][next_state]
            base = costs[t] + cont # shape (S, A)
            #q = base@A
            q = Optimisation.um_break_4_step_break_error(config,config.pc.control_trust ,base)
            # finds the min element along each row. 
            policy[t] = np.argmin(q, axis=1)
            V[t] = np.min(q, axis=1)
        return policy,V

    staticmethod
    def extract_optimal_trajectory( config:configs.config,
                                   policy,V):
        gc = config.gc
        tc = config.tc
        N_TIME = len(policy)
        #l0 = int(N_LANES/2)
        l0 = gc.N_LANES-2
        s0 = 0
        d0 = 0
        optmal_path = np.full((N_TIME,12), 0.0)
        for t in range(N_TIME):
            p0 = tc.get_index_from_policy_tuple(l0,s0,d0)
            #print(index)
            cost = V[t][p0]
            p1 = policy[t][p0]
            l1,s1,d1 = tc.get_policy_tuple_from_index(p1)
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
            optmal_path[t,11] = 0.5*(gc.index_to_speed(s0) + gc.index_to_speed(s1))
            
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

    staticmethod
    def add_track_data_to_optimal_trajectory(   config:configs.config,
                                                optimal_trajectory, 
                                                opt_data):
        gc = config.gc
        pc = config.pc
        aot_df = optimal_trajectory
        #print(f"Test that the frames have the same length {len(aot_df)} {len(opt_data)}")

        N_TIME = len(aot_df)
        #import track data
        x_coord_arr = np.full((gc.N_LANES,N_TIME), pc.MAX_COST_VALUE)
        y_coord_arr = np.full((gc.N_LANES,N_TIME), pc.MAX_COST_VALUE)

        for (l) in range(gc.N_LANES):
            x_coord_arr[l] = opt_data[f"x_lane{l}"].to_numpy()
            y_coord_arr[l] = opt_data[f"y_lane{l}"].to_numpy()
        x_coord_arr_tr = x_coord_arr.transpose()
        y_coord_arr_tr = y_coord_arr.transpose()
        def get_x_coord(t,l):
            return x_coord_arr_tr[int(t),int(l)]
        def get_y_coord(t,l):
            return y_coord_arr_tr[int(t),int(l)]

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

        cols_to_add = ["x_lane" + str(index) for index in range(gc.N_LANES)] + ["y_lane" + str(index) for index in range(gc.N_LANES)]
        aot_df = aot_df.hstack(
            opt_data.select(cols_to_add)
        )


        #Adds forces used in the topmisation
        traj = aot_df
        x_arr = np.array(list(traj["tr_x"]))
        y_arr = np.array(traj["tr_y"])
        l0_arr = np.array(traj["l0"])
        l1_arr = np.array(traj["l1"])
        s0_arr = np.array(traj["s0"])
        s1_arr = np.array(traj["s1"])
        d0_arr = np.array(traj["d0"])
        d1_arr = np.array(traj["d1"])
        debug = False

        N_TIME = len(x_coord_arr_tr)
        forces = np.full((N_TIME,9), 0.0)
        eps=1e-12

        def get_kappa(x00,x0,x1,y00,y0,y1):
            a = ((x0 - x1)**2 + (y0 - y1)**2)**0.5
            b = ((x00 - x1)**2 + (y00 - y1)**2)**0.5
            c = ((x00 - x0)**2 + (y00 - y0)**2)**0.5
            # signed twice-area via cross product
            area2 = (x0 - x00) * (y1 - y00) - (x1 - x00) * (y0 - y00)
            A = abs(area2) * 0.5
            den = a * b * c
            num = 4.0 * A
            kappa = num / den if den > eps else 0.0
            return kappa
        
        def get_balanced_kappa(index):
            l00 = l0_arr[index-1]
            l0 = l0_arr[index]
            l1 = l0_arr[index+1]
            x00 = x_coord_arr[l00][index-1]
            x0 =  x_coord_arr[l0][index]
            x1 =  x_coord_arr[l1][index+1]
            y00 = y_coord_arr[l00][index-1]
            y0 =  y_coord_arr[l0][index]
            y1 =  y_coord_arr[l1][index+1]
            kappa = get_kappa(x00,x0,x1,y00,y0,y1)
            if l00 == l0 == l1:
                return kappa
            else:
                x00 = x_coord_arr[l00][index-1]
                x0 =  x_coord_arr[l00][index]
                x1 =  x_coord_arr[l00][index+1]
                y00 = y_coord_arr[l00][index-1]
                y0 =  y_coord_arr[l00][index]
                y1 =  y_coord_arr[l00][index+1]
                kappa00 = get_kappa(x00,x0,x1,y00,y0,y1)
                x00 = x_coord_arr[l0][index-1]
                x0 =  x_coord_arr[l0][index]
                x1 =  x_coord_arr[l0][index+1]
                y00 = y_coord_arr[l00][index-1]
                y0 =  y_coord_arr[l0][index]
                y1 =  y_coord_arr[l0][index+1]
                kappa11 = get_kappa(x00,x0,x1,y00,y0,y1)                
                x00 = x_coord_arr[l1][index-1]
                x0 =  x_coord_arr[l1][index]
                x1 =  x_coord_arr[l1][index+1]
                y00 = y_coord_arr[l1][index-1]
                y0 =  y_coord_arr[l1][index]
                y1 =  y_coord_arr[l1][index+1]
                kappa22 = get_kappa(x00,x0,x1,y00,y0,y1)
                if kappa < kappa00 and kappa < kappa11 and kappa < kappa22: 
                    return kappa 
                kappaMax = 0.0
                if kappa00 > kappa11 and kappa00 > kappa22: kappaMax = kappa00
                elif kappa11 > kappa22: kappaMax = kappa11
                else: kappaMax = kappa22
                kappa_balanced = kappaMax + 0.5*(kappa-kappaMax)
                return kappa_balanced

        limit_trust = config.pc.limit_trust
        for index in range(0,N_TIME-1):
            x0 = x_arr[index]
            x1 = x_arr[index+1]
            y0 = y_arr[index]
            y1 = y_arr[index+1]
            d = ((x1-x0)**2 + (y1-y0)**2)**0.5
            s0 = config.gc.index_to_speed(s0_arr[index])
            s1 = config.gc.index_to_speed(s1_arr[index])
            av_speed = 0.5*(s0 + s1)
            av_speed2 = av_speed**2 
            time_cost = d / av_speed
            delta_speed = s1 - s0
            acc = delta_speed/time_cost if time_cost > eps else 0.0
            kappa = 0.0
            if index > 0:
                kappa = get_balanced_kappa(index)
                # x00 = x_arr[index-1]
                # y00 = y_arr[index-1]
                # kappa = get_kappa(x00,x0,x1,y00,y0,y1)
                # a = ((x0 - x1)**2 + (y0 - y1)**2)**0.5
                # b = ((x00 - x1)**2 + (y00 - y1)**2)**0.5
                # c = ((x00 - x0)**2 + (y00 - y0)**2)**0.5
                # # signed twice-area via cross product
                # area2 = (x0 - x00) * (y1 - y00) - (x1 - x00) * (y0 - y00)
                # A = abs(area2) * 0.5
                # den = a * b * c
                # num = 4.0 * A
                # kappa = num / den if den > eps else 0.0
            speed2 = s0**2
            a_asym = abs(acc) if acc > 0.0 else abs(config.pc.F_X_neg_scale*acc)
            F_acc = config.pc.M * a_asym + config.pc.tau_h*av_speed2
            F_acc_ef = config.pc.Power/max(av_speed, 10.0)
            acc_cost = config.pc.M * a_asym
            friction_cost = config.pc.tau_h*av_speed2
            F_x = config.pc.M*abs(a_asym)*config.pc.F_X_scale
            F_y = config.pc.M*kappa*speed2*config.pc.F_Y_scale
            F_z = config.pc.mu*(config.pc.M*config.pc.g + config.pc.tau_v*speed2)
            F_t = (F_x**2 + F_y**2) - limit_trust*limit_trust*(F_z**2)
            
            #make cost
            cost = time_cost if ((F_acc < limit_trust*config.pc.Power/av_speed) and (F_t < 0.0)) else config.pc.MAX_COST_VALUE
            forces[index][0] = F_acc
            forces[index][1] = F_x
            forces[index][2] = F_y   
            forces[index][3] = F_z   
            forces[index][4] = F_t
            forces[index][5] = cost
            forces[index][6] = acc_cost
            forces[index][7] = friction_cost
            forces[index][8] = F_acc_ef
            
            if debug:
                l0 = l0_arr[index]
                l1 = l1_arr[index]
                s0 = s0_arr[index]
                s1 = s1_arr[index]
                d0 = d0_arr[index]
                d1 = d1_arr[index]

                acc_test = opt_data[f"acc_l{l0}s{s0}_l{l1}s{s1}"][index]
                kappa_test = opt_data[f"ka_l{l0}d{d0}_l{l1}d{d1}"][index]
                F_acc_test = opt_data[f"Facc_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}"][index]
                F_x_test = opt_data[f"Fx_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}"][index]
                F_y_test = opt_data[f"Fy_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}"][index]
                F_z_test = opt_data[f"Fz_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}"][index]
                if index < 150:
                    print(f"acc:{acc}/{acc_test} ka:{kappa}/{kappa_test} F_acc:{F_acc}/{F_acc_test} F_x:{F_x}/{F_x_test} F_y:{F_y}/{F_y_test} F_z:{F_z}/{F_z_test}")


        forces_t = forces.transpose()
        traj = traj.with_columns(pl.Series(name="F_acc", values=forces_t[0]))
        traj = traj.with_columns(pl.Series(name="Acc_cost", values=forces_t[6]))
        traj = traj.with_columns(pl.Series(name="Friction_cost", values=forces_t[7]))
        traj = traj.with_columns(pl.Series(name="F_acc_ef", values=forces_t[8]))
        traj = traj.with_columns(pl.Series(name="F_x", values=forces_t[1]))
        traj = traj.with_columns(pl.Series(name="F_y", values=forces_t[2]))
        traj = traj.with_columns(pl.Series(name="F_z", values=forces_t[3]))
        traj = traj.with_columns(pl.Series(name="F_t", values=forces_t[4]))
        traj = traj.with_columns(pl.Series(name="cost_O", values=forces_t[5]))
        return traj

    staticmethod
    def make_diagnostic_plots(  gc:configs.config,
                                df, show_plots, run_path):
        print("***** Track plots *****")
        vis.make_track_plot_2(gc, df, show_plots, run_path + r"\\Trajectory.html")


        print("***** Plots with speed and power usage *****")
        x_data = df["index"].to_numpy()
        s_data = df["speed"].to_numpy()
        vis.pl_row_plots(x_data, {"speed":s_data}, "speed", show_plots, run_path + r"\\Speed.html")
        
        # F_acc_data = df["F_acc"].to_numpy()
        # Power_data = np.full(len(F_acc_data),gc.pc.Power)
        # vis.pl_row_plots(x_data, {"F_acc":F_acc_data,"MAX_FORCE":Power_data}, "Acc force Usage", show_plots, run_path + r"\\F_acc.html")
        
        F_acc_data = df["F_acc_ef"].to_numpy().clip(0,50000)
        Friction_cost = df["Friction_cost"].to_numpy()
        Acc_cost = df["Acc_cost"].to_numpy()
        vis.pl_row_plots(x_data, {"F_acc_ef":F_acc_data,"Friction_cost":Friction_cost,"Acc_cost":Acc_cost}, "Acc force Usage", show_plots, run_path + r"\\F_acc.html")
        
        F_x_data = df["F_x"].to_numpy()
        F_y_data = df["F_y"].to_numpy()
        F_z_data = df["F_z"].to_numpy()
        vis.pl_row_plots(x_data, {"F_x":F_x_data,"F_y":F_y_data,"F_z":F_z_data}, "Force Usage", show_plots, run_path + r"\\F_usage.html")
        #vis.pl_row_plots(x_data, {"F_x*F_x":F_x_data*F_x_data,"F_y*F_y":F_y_data*F_y_data,"F_z*F_z":F_z_data*F_z_data}, "Force Usage", show_plots, run_path + r"\\F2_usage.html")
        return

    staticmethod
    def print_opt_result(config:configs.config,df):
        gc = config.gc
        tot_cost = max(df["tot_cost"])
        tot_cost_robust = sum(list(df["times_est"]))
        max_speed = max(df["s0"])*gc.INDEX_SPEED_MULTIPLIER
        print(f"tot_cost:{tot_cost} tot_cost_robust={tot_cost_robust} max_speed={max_speed}")
        result_df = pl.DataFrame({
            "tot_cost":tot_cost,
            "tot_cost_robust":tot_cost_robust,
            "max_speed":max_speed
        })
        print("")
        return result_df

    staticmethod
    def run_stack(  config:configs.config,costs_df):
        control_trust = config.pc.control_trust
        #run_path, needs_to_run = book_keeping.init_test_run_folder(configs.folder_config.RUN_FOLDER,gc.N_LANES,gc.N_SPEEDS,gc.N_DIRECTIONS,MASS,g,tau_h,tau_v,mu,Power,F_X_neg_scale,F_X_scale,F_Y_scale,rho)
        run_path, needs_to_run = book_keeping.init_test_run_folder(config, configs.folder_config.RUN_FOLDER)
        #print(f"PARAMETERS: M:{MASS} g:{g} tau_h:{tau_h} tau_v:{tau_v} mu:{mu} Power:{Power} F_X_neg_scale:{F_X_neg_scale} F_X_scale:{F_X_scale} F_Y_scale:{F_Y_scale} rho:{rho} ")
        print("***** making cost_arr *****")
        #costs,_ = force_frame.np_action_range_frames_for_name(costs_df,"cost",save=False)
        costs = force_frame.Forces.pl_to_np(config,costs_df,"cost")
        print(f"***** optimal policy with uncertainty {control_trust} *****")
        if control_trust == 1.0:
            policy,V = Optimisation.backward_optimiser(costs)
        else:
            policy,V = Optimisation.backward_optimiser_with_uncertainty(config,costs)
        print("***** extracting optimal trajectory *****")
        optimal_trajectory_df = Optimisation.extract_optimal_trajectory(config,policy,V)
        print("***** augmenting trajectory with track data *****")
        traj_df = Optimisation.add_track_data_to_optimal_trajectory(config,optimal_trajectory_df,costs_df)
        #return traj_df
        # print("***** adding force terms to trajectory *****")
        # atraj_df = Optimisation.add_Forces_data_to_optimal_trajectory(traj_df, acc_arr, av_arr, MASS, g, tau_h, tau_v, mu, Power, F_X_neg_scale, F_X_scale, F_Y_scale)
        print("*****making plots *****")
        #path_opt.make_diagnostic_plots(atraj_df)
        book_keeping.save_frame_to_run_dir(traj_df, run_path, "trajectory")
        print("***** make result frame *****")
        results_frame = Optimisation.print_opt_result(config,traj_df)
        print("***** making plots *****")
        Optimisation.make_diagnostic_plots(config,traj_df, True, str(run_path))
        book_keeping.save_frame_to_run_dir(results_frame, run_path, "results")
        return traj_df, V
    
    

