import pickle

import hsr


class WaterEnv(hsr.HSREnv):
    cup_regions = [None, 1, 2, 3, 4, 5, 6, 7, 8]
    cup_nos = [None, 1]
    station = [None, 10]
    def init_arg(self, task_name):
        return [0]

