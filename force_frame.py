import polars as pl
import numpy as np
import os
from pathlib import Path

import Utils.config as configs

class Forces:
 
    staticmethod
    def add_Forces_and_constraints( config: configs.config,
                                    ldf:pl.DataFrame):
        gc = config.gc
        tc = config.tc

        M = config.pc.M
        g = config.pc.g
        tau_h = config.pc.tau_h
        tau_v = config.pc.tau_v
        mu = config.pc.mu
        Power = config.pc.Power
        F_X_neg_scale = config.pc.F_X_neg_scale
        F_X_scale = config.pc.F_X_scale
        F_Y_scale = config.pc.F_Y_scale
        limit_trust = config.pc.limit_trust
        print(f"M:{M} M:{g} tau_h:{tau_h} tau_v:{tau_v} mu:{mu} Power:{Power}  F_X_neg_scale:{F_X_neg_scale} F_X_scale:{F_X_scale} F_Y_scale:{F_Y_scale} limit_trust:{limit_trust}")

        all_cols: list[pl.Expr] = []
        for (l0,s0,d0,l1,s1,d1) in tc.action_range:
            #data
            a = pl.col(f"acc_l{l0}s{s0}_l{l1}s{s1}")
            #original
            #kappa = pl.col(f"ka_l{l0}d{d0}_l{l1}d{d1}")
            #No lane changes
            #kappa = pl.when(l0==l1).then(pl.col(f"ka_l{l0}d{d0}_l{l1}d{d1}")).otherwise(config.pc.Power)
            #tempered
            if l0 == l1:
                kappa = pl.col(f"ka_l{l0}d{d0}_l{l1}d{d1}")
            else:
                l00 = config.tc.get_previous_lane(l0,d0)
                kappa00 = pl.col(f"ka_l{l00}d{0}_l{l00}d{0}")
                kappa11 = pl.col(f"ka_l{l0}d{0}_l{l0}d{0}")
                kappa22 = pl.col(f"ka_l{l1}d{0}_l{l1}d{0}")
                kappa0 = pl.when((kappa00>kappa11) & (kappa00>kappa22)).then(kappa00).otherwise(pl.when((kappa11>kappa22)).then(kappa11).otherwise(kappa22))
                kappa1 = pl.col(f"ka_l{l0}d{d0}_l{l1}d{d1}")
                kappa_balanced = kappa0 + 0.5*(kappa1-kappa0)
                kappa = kappa_balanced

            #kappa = pl.col(f"ka_l{l0}d{d0}_l{l1}d{d1}")

            #capped
            # mult = 5.0
            # kappa_capped = pl.when(kappa_balanced < mult*kappa0).then(kappa_balanced).otherwise(mult*kappa0)
            #kappa = pl.when(kappa1 < kappa0).then(kappa1).otherwise(kappa_capped) 
            
            #kappa = pl.col(f"ka_l{l0}d{0}_l{l1}d{0}")
            M_lit = pl.lit(M)
            Mg_lit = pl.lit(M*g)
            tau_h_lit = pl.lit(tau_h)
            tau_v_lit = pl.lit(tau_v)
            F_x_scale_lit = pl.lit(F_X_scale)
            F_X_neg_scale_lit = pl.lit(F_X_neg_scale)
            F_y_scale_lit = pl.lit(F_Y_scale)
            limit_trust = pl.lit(config.pc.limit_trust)
            #assembly
            speed = gc.index_to_speed(s0) 
            speed2 = speed*speed
            av_speed = 0.5*(gc.index_to_speed(s0) + gc.index_to_speed(s1))
            av_speed2 = av_speed*av_speed
            a_asym = (pl.when(a>0).then(a).otherwise(F_X_neg_scale_lit*a)).abs()
            F_acc = M_lit*a_asym + tau_h_lit*av_speed2
            F_x = M_lit*abs(a_asym)*F_x_scale_lit #could make this a
            F_y = M_lit*kappa*speed2*F_y_scale_lit
            F_z = pl.lit(mu)*(Mg_lit + tau_v_lit*speed2)
            F_t = (F_x*F_x + F_y*F_y) - limit_trust*limit_trust*(F_z*F_z)
            #make cost
            tcost = pl.col(f"tc_l{l0}s{s0}_l{l1}s{s1}")
            cost = pl.when((F_acc < limit_trust*Power/av_speed) & (F_t < 0.0)).then(tcost).otherwise(pl.lit(config.pc.Power))
            cost_col = cost.alias(f"cost_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}")
            all_cols.extend([cost_col])
            
        return ldf.with_columns(all_cols)



    staticmethod
    def pl_to_np(   config:configs.config,
                    df:pl.LazyFrame, 
                    name): 
                    #save=True):
        gc = config.gc
        tc = config.tc
        pc = config.pc
        N_TIME = len(df)
        cost_dim = gc.N_LANES*gc.N_SPEEDS*gc.N_DIRECTIONS
        arr = np.full((cost_dim,cost_dim,N_TIME), pc.MAX_COST_VALUE)
        def name_fun(l0,s0,d0,l1,s1,d1):
            return f"{name}_l{l0}s{s0}d{d0}_l{l1}s{s1}d{d1}"
        for (l0,s0,d0,l1,s1,d1) in tc.action_range:
            index0 = tc.get_index_from_policy_tuple(l0,s0,d0) 
            index1 = tc.get_index_from_policy_tuple(l1,s1,d1)
            arr[index0][index1] = df[name_fun(l0,s0,d0,l1,s1,d1)].to_numpy()
        arr = arr.transpose(2,0,1)
        # path = None
        # if save:
        #     path = Path(gc.DATA_FOLDER + f"\\{name}_{gc.N_LANES}")
        #     np.save(path, arr)
        return arr

