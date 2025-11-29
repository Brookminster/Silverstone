FOLDER = "C:\\Users\\johan//OneDrive//Documents//Python//Python//SilverStone"
RUN_FOLDER = "C:\\Users\\johan//OneDrive//Documents//Python//Python//SilverStone//Runs"
DATA_FOLDER = "C:\\Users\\johan//OneDrive//Documents//Python//Python//SilverStone//Data"
test = False
if test:
    N_TIME_LIMIT = 500
    N_LANES = 1
    N_SPEEDS = 20
    N_DIRECTIONS = 3
    N_SPEED_DEVIATION_RED = 5
    N_SPEED_DEVIATION_INC = 3
else:
    #State space
    N_TIME_LIMIT = 500
    N_LANES = 1 #to cope well with the min in optimisation, lane 0 has to be on the inside. This gives a prior on preference
    N_SPEEDS = 30 #39
    N_DIRECTIONS = 3  #to cope well with the min in optimisation, 0 means stay in lane, 1 means move inwards, 2 means move out. This gives a prior on preference
    N_SPEED_DEVIATION_RED = 4
    N_SPEED_DEVIATION_INC = 3

#Action space
N_LANE_DEVIATION=int(N_DIRECTIONS/2)
INDEX_SPEED_MULTIPLIER = 3.0 #200 mph is about 90 m/s. We have 20 steps. 

#Motion

def index_to_speed(index): 
    speed = (index)*INDEX_SPEED_MULTIPLIER 
    if speed < 1.0: return 1.0
    return speed
# import numpy as np
# speed_lists = [int(s) for s in [1, 5, 10, 15] + list(np.linspace(20, 60, 20, endpoint=False)) + list(np.linspace(60, 90, 15, endpoint=False))]
# def index_to_speed(index): 
#     return speed_lists[index]
MAX_COST_VALUE = 100000.0
