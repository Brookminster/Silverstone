import polars as pl
import numpy as np
import os
from pathlib import Path

import Utils.constants
from Utils.constants import FOLDER
from Utils.constants import N_LANES, N_SPEEDS, N_DIRECTIONS, N_SPEED_DEVIATION_RED, N_SPEED_DEVIATION_INC, N_LANE_DEVIATION, N_TIME_LIMIT
from Utils.constants import MAX_COST_VALUE
from Utils.constants import index_to_speed

#lane transitions
def l0_range():
    return range(N_LANES)
def l1_range(index):
    return range(max(0,index-N_LANE_DEVIATION), min(N_LANES,index+N_LANE_DEVIATION+1))
distance_pairs = [(l0,l1) for l0 in l0_range() for l1 in l1_range(l0)]
def test_distance_pairs():
    print("Distance pairs")
    for (l0,l1) in distance_pairs:
        print(f"{l0} -> {l1}")
    return 

# returns the possible directions for the previous step
def d0_range(l0):
    if (l0 == 0) and (l0 == N_LANES-1): return [0]
    if l0 == 0: return [0,1]
    if l0 == N_LANES-1: return [0,2]
    return [0,1,2]

# returns the possible directions for this step
def d1_range(l0):
    if (l0 == 0) and (l0 == N_LANES-1): return [0]
    if l0 == 0: return [0,2]
    if l0 == N_LANES-1: return [0,1]
    return [0,1,2]
def get_previous_lane(l0,d0):
    if d0 == 1: return l0 + 1
    if d0 == 2: return l0 - 1
    return l0 
def get_next_lane(l0,d0):
    if d0 == 1: return l0 - 1
    if d0 == 2: return l0 + 1
    return l0 

def s0_range():
    return range(N_SPEEDS)
def s1_range(index):
    return range(max(0,index-N_SPEED_DEVIATION_RED), min(N_SPEEDS,index+N_SPEED_DEVIATION_INC+1))

#speed_ranges
speeds_range = [(s0,s1) for s0 in s0_range() for s1 in s1_range(s0)]
#lanes speed range
lanes_dir_range = [(l0,d0,get_next_lane(l0,d1),d1) for l0 in l0_range() for d0 in d0_range(l0) for d1 in d1_range(l0) ]
#lanes speed range
lanes_speed_range = [(l0,s0,l1,s1) for l0 in l0_range() for l1 in l1_range(l0) for s0 in s0_range() for s1 in s1_range(s0)]

#state*action space
action_range = [(l0,s0,d0,get_next_lane(l0,d1),s1,d1) for l0 in l0_range() for s0 in s0_range() for d0 in d0_range(l0) for s1 in s1_range(s0) for d1 in d1_range(l0) ]
def test_action_range():
    print("state by action space", len(action_range))
    for (l0,s0,d0,l1,s1,d1) in action_range:
        print(f"l{l0} s{s0} d{d0} -> l{l1} s{s1} d{d1}")
    return 

def get_index_from_policy_tuple(l0,s0,d0):
    return l0*N_SPEEDS*N_DIRECTIONS+s0*N_DIRECTIONS+d0

def get_policy_tuple_from_index(index):
    l = int(index/(N_SPEEDS*N_DIRECTIONS))
    s = int((index-l*(N_SPEEDS*N_DIRECTIONS))/N_DIRECTIONS)
    d = int(index)%N_DIRECTIONS
    return (l,s,d)

def get_index_from_ls_double(l0,s0):
    return l0*N_SPEEDS+s0

def get_index_from_ld_double(l0,s0):
    return l0*N_DIRECTIONS+s0