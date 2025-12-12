class folder_config:
    TRACK_FOLDER = r"C:\Users\johan\OneDrive\Documents\Python\Python\SilverStone\TrackData"
    FOLDER = r"C:\Users\johan\OneDrive\Documents\Python\Python\SilverStone"
    RUN_FOLDER = r"C:\Users\johan\OneDrive\Documents\Python\Python\SilverStone\Runs"
    DATA_FOLDER = r"C:\Users\johan\OneDrive\Documents\Python\Python\SilverStone\Data"


class grid_config:
    def __init__(
        self,
        n_lanes=11,
        n_speeds=30,
        n_time_limit=3000,
        n_directions=3,
        n_speed_deviation_red=5,
        n_speed_deviation_inc=2,
        index_speed_multiplier=3.0
    ):
        # Base values
        self.N_TIME_LIMIT = n_time_limit    #max number of steps in the grid
        self.N_LANES = n_lanes              #number of parallel lanes on the track
        self.N_SPEEDS = n_speeds            #number of speed steps
        self.N_DIRECTIONS = n_directions    #number of possible turns for the car (straight, inward, outward)
        self.N_SPEED_DEVIATION_RED = n_speed_deviation_red      #maximal acceleration
        self.N_SPEED_DEVIATION_INC = n_speed_deviation_inc      #maximal breaking speed
        self.INDEX_SPEED_MULTIPLIER = index_speed_multiplier    #translantion from speed grid to speed

        # Derived values
        self.N_LANE_DEVIATION = int(self.N_DIRECTIONS / 2)

    @classmethod
    def test_mode(
            cls, 
            n_time_limit=300,
            n_lanes=11,
            n_speeds=30,
            n_directions=3,
            n_speed_deviation_red=5,
            n_speed_deviation_inc=3):
        return cls(
            n_time_limit = n_time_limit,
            n_lanes=n_lanes,
            n_speeds=n_speeds,
            n_directions=n_directions,
            n_speed_deviation_red=n_speed_deviation_red,
            n_speed_deviation_inc=n_speed_deviation_inc,
        )

    @classmethod
    def standard_mode(cls):
        return cls(
            n_time_limit=3000,
            n_lanes=11,
            n_speeds=30,
            n_directions=3,
            n_speed_deviation_red=5,
            n_speed_deviation_inc=3,
        )
    
    # Motion
    def index_to_speed(self, index: int) -> float:
        speed = index * self.INDEX_SPEED_MULTIPLIER
        return max(speed, 1.0)
    



class trajectory_config:

    def __init__(self, gc:grid_config):
        self.gc = gc
        self.N_LANES = gc.N_LANES
        self.N_LANE_DEVIATION = gc.N_LANE_DEVIATION
        self.N_SPEEDS = gc.N_SPEEDS
        self.N_SPEED_DEVIATION_RED = gc.N_SPEED_DEVIATION_RED
        self.N_SPEED_DEVIATION_INC = gc.N_SPEED_DEVIATION_INC
        self.N_DIRECTIONS = gc.N_DIRECTIONS

        # Precompute the main ranges used repeatedly
        self.distance_pairs = [
            (l0, l1)
            for l0 in self.l0_range()
            for l1 in self.l1_range(l0)
        ]

        self.speeds_range = [
            (s0, s1)
            for s0 in self.s0_range()
            for s1 in self.s1_range(s0)
        ]

        # (l0, d0, l1, d1) where l1 is next lane from d1
        self.lanes_dir_range = [
            (l0, d0, self.get_next_lane(l0, d1), d1)
            for l0 in self.l0_range()
            for d0 in self.d0_range(l0)
            for d1 in self.d1_range(l0)
        ]

        # (l0, s0, l1, s1)
        self.lanes_speed_range = [
            (l0, s0, l1, s1)
            for l0 in self.l0_range()
            for l1 in self.l1_range(l0)
            for s0 in self.s0_range()
            for s1 in self.s1_range(s0)
        ]

        # (l0, s0, d0, l1, s1, d1)
        self.action_range = [
            (l0, s0, d0, self.get_next_lane(l0, d1), s1, d1)
            for l0 in self.l0_range()
            for s0 in self.s0_range()
            for d0 in self.d0_range(l0)
            for s1 in self.s1_range(s0)
            for d1 in self.d1_range(l0)
        ]

    # -------- lane ranges --------
    def l0_range(self):
        return range(self.N_LANES)

    def l1_range(self, index):
        return range(
            max(0, index - self.N_LANE_DEVIATION),
            min(self.N_LANES, index + self.N_LANE_DEVIATION + 1),
        )

    def test_distance_pairs(self):
        print("Distance pairs")
        for (l0, l1) in self.distance_pairs:
            print(f"{l0} -> {l1}")

    # -------- direction ranges --------
    # returns the possible directions for the previous step
    def d0_range(self, l0):
        if (l0 == 0) and (l0 == self.N_LANES - 1):
            return [0]
        if l0 == 0:
            return [0, 1]
        if l0 == self.N_LANES - 1:
            return [0, 2]
        return [0, 1, 2]

    # returns the possible directions for this step
    def d1_range(self, l0):
        if (l0 == 0) and (l0 == self.N_LANES - 1):
            return [0]
        if l0 == 0:
            return [0, 2]
        if l0 == self.N_LANES - 1:
            return [0, 1]
        return [0, 1, 2]

    def get_previous_lane(self, l0, d0):
        if d0 == 1:
            return l0 + 1
        if d0 == 2:
            return l0 - 1
        return l0

    def get_next_lane(self, l0, d0):
        if d0 == 1:
            return l0 - 1
        if d0 == 2:
            return l0 + 1
        return l0

    # -------- speed ranges --------
    def s0_range(self):
        return range(self.N_SPEEDS)

    def s1_range(self, index):
        return range(
            max(0, index - self.N_SPEED_DEVIATION_RED),
            min(self.N_SPEEDS, index + self.N_SPEED_DEVIATION_INC + 1),
        )

    # -------- debugging helpers --------
    def test_action_range(self):
        print("state by action space", len(self.action_range))
        for (l0, s0, d0, l1, s1, d1) in self.action_range:
            print(f"l{l0} s{s0} d{d0} -> l{l1} s{s1} d{d1}")

    # -------- index <-> tuple helpers --------
    def get_index_from_policy_tuple(self, l0, s0, d0):
        return (
            l0 * self.N_SPEEDS * self.N_DIRECTIONS
            + s0 * self.N_DIRECTIONS
            + d0
        )

    def get_policy_tuple_from_index(self, index):
        ls_block = self.N_SPEEDS * self.N_DIRECTIONS
        l = index // ls_block
        s = (index - l * ls_block) // self.N_DIRECTIONS
        d = index % self.N_DIRECTIONS
        return (l, s, d)

    def get_index_from_ls_double(self, l0, s0):
        return l0 * self.N_SPEEDS + s0

    def get_index_from_ld_double(self, l0, d0):
        return l0 * self.N_DIRECTIONS + d0
    


class physics_config:
    def __init__(
        self,
        M,                  # mass [kg]
        g,                  # gravity [m/s^2]
        tau_h,              # horizontal air resistance
        tau_v,              # vertical air resistance
        mu,                 # friction coefficient
        Power,              # horsepower
        F_X_neg_scale,      # scale for braking (negative X force)
        F_X_scale,          # scale for max positive X force
        F_Y_scale,          # scale for max lateral force
        limit_trust=0.0,    # confidence in getting close to the limits
        control_trust=0.0,  # confidence in the control of the car
        max_cost_value=100000.0
    ):
        # Base values
        self.M = M
        self.g = g
        self.tau_h = tau_h
        self.tau_v = tau_v
        self.mu = mu
        self.Power = Power
        self.F_X_neg_scale = F_X_neg_scale
        self.F_X_scale = F_X_scale
        self.F_Y_scale = F_Y_scale
        self.WEIGHT = self.M * self.g 
        self.limit_trust = limit_trust
        self.control_trust = control_trust
        self.MAX_COST_VALUE=max_cost_value

    # Second constructor (factory)
    @classmethod
    def standard_mode(cls):
        return cls(
            M=800.0,
            g=9.81,
            tau_h=0.62,
            tau_v=5.4,
            mu=1.0,
            Power=25000.0,
            F_X_neg_scale=0.3,
            F_X_scale=0.8, #0.5,
            F_Y_scale=1.0,
            limit_trust = 0.0,
            control_trust = 0.0
        )
    
    #model with the extra speed term scaling force
    # P/v > m*a + tau_h*v^2
    # a = delta(v)/delta(t) = (v2 - u2)/(2 delta(d))
    # P/v -  tau_h*v^2 > m*a
    # P/v -  tau_h*v^2 > 0.9*m*(v2 - u2)/(2 delta(d))
    # P -  tau_h*v^3 > 0.9*m*(v2 - u2)/(2 delta(d))*v
    # (P - 0.9*m*v*(v2 - u2)/(2 delta(d)))/v^3 = tau_h 
    # in the limit v_max = 90.0, a = delta(v)/delta(d) = 3/10
    # test
    # hp = 1000
    # watts = hp*750
    # gc = configs.grid_config.standard_mode()
    # v1 = (gc.N_SPEEDS-1)*gc.INDEX_SPEED_MULTIPLIER
    # v2 = (gc.N_SPEEDS-2)*gc.INDEX_SPEED_MULTIPLIER
    # v3 = (gc.N_SPEEDS-3)*gc.INDEX_SPEED_MULTIPLIER
    # v_av1 = 0.5*(v1+v2)
    # v_av2 = 0.5*(v2+v3)
    # m = 800.0
    # delta_dist = 15
    # t1 = m*(v1*v1 - v2*v2)/(2*delta_dist)*v_av1
    # t2 = m*(v2*v2 - v3*v3)/(2*delta_dist)*v_av2
    # print(watts)
    # print(t1)
    # print(t2)
    # print(v_av2*v_av2*v_av2)
    # print( (t1+ v_av2*v_av2*v_av2)/ watts)
    #means, we need about 2300 PS to get to full speed.
    @staticmethod
    def modelV1(gc,horse_power):
        power = horse_power*750
        tau_h = 2.0 
        return physics_config(
            M=800.0,
            g=9.81,
            tau_h=tau_h,
            tau_v=5.4,#5.4,
            mu=1.0,
            Power=power,
            F_X_neg_scale=0.5,
            F_X_scale=1.0,
            F_Y_scale=1.0,
            limit_trust = 1.0,
            control_trust = 1.0
        )
    


class config:
    def __init__(
        self,
        grid_config,
        trajectory_config,
        physics_config
    ):
        # Base values
        self.gc = grid_config
        self.tc = trajectory_config
        self.pc = physics_config

    @classmethod
    def standard_config(cls):
        gc = grid_config.standard_mode()
        tc = trajectory_config(gc)
        pc = physics_config.standard_mode()
        return cls(gc,tc,pc)
    
    @staticmethod
    def modelV1(power,n_lanes=11):
        gc = grid_config(n_lanes=n_lanes)
        tc = trajectory_config(gc)
        pc = physics_config.modelV1(gc,power)
        return config(gc,tc,pc)