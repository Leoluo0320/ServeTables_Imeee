import math
import pickle
import gazebo_getpos
import hierarchical
import utils
from envs.water import WaterEnv

# think about what model for each skill.
DEBUG = False
PRETRAINED = True
V = 0.07
OMEGA = math.pi / 5.
STATION_POSITION = [0.75, -2]
FAKE_WATER_LEVEL = 2
ARM_STAND_BY = [0, -0.6, -1.57, -1.57, 0.6]


"""Upper Level Skills"""


class ServeTables(hierarchical.Skill):
    # these variables are used for learning
    # Table: task execute in fix sequence
    arg_in_len = 1
    sub_skill_names = ['ServeTable', 'MoveArm', 'MoveGripper']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [consta]
        ret_val:
        """
        [consta] = arg
        if cnt == 0:
            return 'MoveGripper', [1.57]
        if cnt == 1:
            return 'MoveArm', [0, -0, -1.57, -1.57, 1.02]
        if cnt == 2:
            return 'ServeTable', None
        else:
            return None, None


class ServeTable(hierarchical.Skill):
    # Table: task execute in fix sequence
    arg_in_len = 0
    sub_skill_names = ['InspectTable', 'FillCups', 'GoStandBy']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: None
        ret_val: region_list
        """
        # region_list = [current_position, region1_cup_no, region2_cup_no, ..., region8_cup_no]
        region_list = [0 for i in range(16)]

        if ret_name == 'InspectTable' or ret_name == 'GoStandBy':   # not necessary, for comprehension
            region_list = ret_val
        return [
            ('InspectTable', None),
            ('GoStandBy', region_list),
            ('FillCups', region_list),
            (None, None)
        ][cnt]


class GoStandBy(hierarchical.Skill):
    arg_in_len = 16
    sub_skill_names = ['MoveBaseAbs']
    ret_out_len = 16

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: region_list
        ret_val: region_list
        """
        # TODO initialize to the region 1 to stand by
        region_list = arg
        region_list[0:8] = [1,0,0,0,0,0,0,0]
        return [['MoveBaseAbs', [0, -1.2, 1.57, V, OMEGA]],
                ['MoveBaseAbs', [0.375, -1.2, 1.57, V, OMEGA]],
                [None, region_list]][cnt]


class InspectTable(hierarchical.Skill):
    # subskill switch between LookAtTable and GoToTable, different ret_val. logpoly2
    arg_in_len = 0
    sub_skill_names = ['GoToTable', 'LookAtTable']
    ret_out_len = 16

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg:
        ret_val: region_list from LookAtTable, Move_succeed from GoToTable
        """

        if ret_name is None:
            # region_list contains 16 integers.
            # current region in one-hot representation and cups number in different region.
            region_list = [0 for i in range(16)]
            # argument: motion_cnt, region_list
            return 'GoToTable', [1] + region_list
        elif ret_name == 'GoToTable':
            # it is return by GoToTable
            return 'LookAtTable', ret_val
        elif ret_name == 'LookAtTable':
            # Return by LookAtTable
            motion_cnt = ret_val[0]
            # region_list = [current_position, region1_cup_no, region2_cup_no, ..., region8_cup_no]
            region_list = ret_val[1:17]
            motion_cnt += 1
            if motion_cnt == 2:
                return 'GoToTable', [motion_cnt] + region_list
            else:
                return None, region_list


class FillCups(hierarchical.Skill):
    # still in a fix sequence, but can change judging condition to others, logpoly2
    model_name = 'log_lin'
    arg_in_len = 16
    sub_skill_names = ['FillCup', 'FillCupsWrapper', 'ScanTable', 'ReturnRegionList']
    ret_out_len = 0
    sub_arg_accuracy = [1e-8 for _ in range(17)]

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg:region list
        ret_val:
        """
        if ret_name == None:
            region_list = arg
        else:
            region_list = ret_val[0:16]

        # when there is no empty cup on the table
        if sum(region_list[8:16]) == 0:
            return None, None
        elif ret_name == None:
            region_list = arg
            return 'ReturnRegionList', region_list
        elif ret_name == 'ReturnRegionList':
            region_list = obs
            return 'FillCupsWrapper', obs
        elif ret_name == 'ScanTable':
            # obs from ScanTable is region list
            region_list = obs
            return 'FillCupsWrapper', obs
        elif ret_name == 'FillCupsWrapper':
            # ret_val from FillCupsWrapper is region_list + [water_level]
            return 'FillCup', ret_val
        elif ret_name == 'FillCup':
            region_list = ret_val
            return 'ScanTable', region_list


class FillCupsWrapper(hierarchical.Skill):
    model_name = 'log_lin'
    arg_in_len = 16
    sub_skill_names = ['EnsureWaterAmount', 'MoveNextRegion', 'SelectNextRegionWrapper']
    ret_out_len = 17

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: region_list
        ret_val:
        """
        region_list = arg[0:16]
        # select destination region, if no empty cups, destin_region = 0

        if ret_name == None:
            return 'EnsureWaterAmount', arg[0:8]
        elif ret_name == 'EnsureWaterAmount':
            [water_level, is_refilled] = ret_val
            return 'SelectNextRegionWrapper', region_list+ret_val
        elif ret_name == 'SelectNextRegionWrapper':
            # ret_val from SelectNextRegion is destination region, water_level, is_refilled.
            curr_region = region_list[0:8]
            return 'MoveNextRegion', curr_region + ret_val
        elif ret_name == 'MoveNextRegion':
            # ret_val from MoveNextRegion is water_level, destination region.
            water_level = ret_val[0]
            destin_region = ret_val[1:9]
            region_list[0:8] = destin_region
            return None, region_list + [water_level]


'''SelectNextRegion and its sub-skills'''

class SelectNextRegionWrapper(hierarchical.Skill):
    model_name = 'log_lin'
    arg_in_len = 18
    sub_skill_names = ['SelectNextRegion']
    ret_out_len = 10
    max_cnt = 16

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: region_list = arg (current region: arg[0:8], region cup number: arg[8:16], [water_level, is_refilled])
        ret_val: destination region in one hot representation
        """
        curr_region = arg[0:8]
        region_cup_num = arg[8:16]
        if ret_name == None:
            return 'SelectNextRegion', curr_region + region_cup_num
        else:
            return None, ret_val[0:8]+arg[16:18]


class SelectNextRegion(hierarchical.Skill):
    model_name = 'log_poly2'
    sub_skill_names = ['ShiftRegion', 'CheckRegion']
    arg_in_len = 16
    ret_out_len = 8
    max_cnt = 16
    '''
    arg_in_len = 18
    ret_out_len = 10
    max_cnt = 16
    '''

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: region_list = arg (current region: arg[0:8], region cup number: arg[8:16], [water_level, is_refilled])
        ret_val: destination region in one hot representation
        """
        curr_region = arg[0:8]
        region_cup_num = arg[8:16]
        # shift one bit should be linear complexity
        # if cnt == 16: # this is not necessary because we will not encounter cases that region list is all 0.
        #     return None, curr_region+arg[16:18]
        if ret_name == None:
            return 'CheckRegion', curr_region+region_cup_num
        elif ret_name == 'ShiftRegion':
            update_region = ret_val
            return 'CheckRegion', ret_val+region_cup_num
        elif ret_name == 'CheckRegion':
            # ret_val from CheckRegion: ret_val[0] cup_num in that region, ret_val[1:9] selected region.
            if ret_val[0]:
                return None, ret_val[1:9]
            else:
                return 'ShiftRegion', ret_val[1:9]


# put these as skills to see if I can train this
class ShiftRegion(hierarchical.Skill):
    model_name = 'log_lin'
    arg_in_len = 8
    sub_skill_names = ['Delay']
    ret_out_len = 8

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: previous region: arg[0:8]
        ret_val: selected region in one hot representation
        """
        prev_region = arg
        index = sum([i*prev_region[i] for i in range(8)])
        prev_region[index] = 0
        prev_region[(index+1)%8] = 1    # this become the next selected region
        return None, prev_region


class CheckRegion(hierarchical.Skill):
    model_name = 'log_poly2'
    arg_in_len = 16
    sub_skill_names = ['Delay']
    ret_out_len = 9

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: region_list = arg, current region: arg[0:8], region cup number: arg[8:16]
        ret_val: destination region in one hot representation
        """
        sele_region = arg[0:8]
        region_cup_num = arg[8:16]
        index = sum([i*sele_region[i] for i in range(8)])
        return None, [region_cup_num[index]]+sele_region


''' skill: MoveNextRegion and its sub-skills'''


class MoveNextRegion(hierarchical.Skill):
    # t-dep logpoly2
    model_name = 'log_poly2'
    arg_in_len = 18
    sub_skill_names = ['MoveBaseAbs', 'MoveAroundTable']
    ret_out_len = 9
    max_cnt = 3  # problematic
    sub_arg_accuracy = [1e-2 for i in range(16)]

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [curr_region, destin_region, water_level, is_refilled]
        ret_val: [water_level]
        """
        curr_region = arg[0:8]
        destin_region = arg[8:16]
        [water_level, is_refilled] = arg[16:18]
        # if use t-model, judgement on isrefilled will be fined, but try log-quadratic
        # in my training set, isrefilled = 0 always, still fail
        if not is_refilled:
            return [('MoveAroundTable', curr_region + destin_region),
                    (None, [water_level]+destin_region)][cnt]
        else:
            # move to region 1 from water station
            curr_region = [1,0,0,0,0,0,0,0]
            return [('MoveBaseAbs', [0.750, -1.5, 1.57, V, OMEGA]),
                    ('MoveAroundTable', curr_region+destin_region),
                    (None, [water_level]+destin_region)][cnt]


class MoveAroundTable(hierarchical.Skill):
    model_name = 'table'
    arg_in_len = 16  # logpoly2
    max_cnt = 2
    sub_skill_names = ['MoveAlongSide', 'MoveToRightSide', 'MoveToLeftSide']
    ret_out_len = 0
    sub_arg_accuracy = [1e-2 for i in range(16)]
    # try different model with construct training set.

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        curr_region = arg[0:8]
        destin_region = arg[8:16]
        if destin_region == [0 for _ in range(8)]:
            destin_region = [1]+[0 for _ in range(7)]
        if cnt == 1:
            return None, None
        # calculate circular linked list index for destination region reference to current regioon: -3,-2,-1,0,1,2,3,4
        num_curr_region = sum([i*curr_region[i] for i in range(8)]) + 1
        num_destin_region = sum([i*destin_region[i] for i in range(8)]) + 1
        if num_destin_region-num_curr_region>4:
            cir_destin_region = num_destin_region-num_curr_region-8
        elif num_destin_region-num_curr_region<-3:
            cir_destin_region = num_destin_region-num_curr_region+8
        else:
            cir_destin_region = num_destin_region-num_curr_region

        # if seperate, decide subskill poly2
        if (1 <= num_curr_region <= 4 and 1 <= num_destin_region <= 4) or \
           (5 <= num_curr_region <= 8 and 5 <= num_destin_region <= 8):  # poly2
            return 'MoveAlongSide', curr_region + destin_region
        # calculate and compare the distance from both direction (treat it as circular linked list).
        elif cir_destin_region > 0:   # destination is on the right side of current location
            return 'MoveToRightSide', curr_region + destin_region
        else:
            return 'MoveToLeftSide', curr_region + destin_region



'''
1, break down to subskill: poly2+poly1
2, keep this way but need poly3, time variant
3, '''
class MoveAlongSide(hierarchical.Skill):
    model_name = 'table'
    arg_in_len = 16  # logpoly2
    sub_skill_names = ['MoveBaseRel']
    ret_out_len = 0
    # try different model with construct training set.

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        curr_region = arg[0:8]
        destin_region = arg[8:16]
        num_curr_region = sum([i*curr_region[i] for i in range(8)]) + 1
        num_destin_region = sum([i*destin_region[i] for i in range(8)]) + 1
        x = 0.4*(num_destin_region-num_curr_region)
        return [('MoveBaseRel', [0, -x, 0, V, OMEGA]),
                (None, None)][cnt]


class MoveToRightSide(hierarchical.Skill):
    model_name = 'table'
    arg_in_len = 16  # t-dep logpoly2
    sub_skill_names = ['MoveBaseRel', 'MoveAlongSide']
    ret_out_len = 0
    max_cnt = 6
    sub_arg_accuracy = [1e-2 for i in range(16)]
    # try different model with construct training set.

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        curr_region = arg[0:8]
        destin_region = arg[8:16]
        num_curr_region = sum([i*curr_region[i] for i in range(8)]) + 1
        num_destin_region = sum([i*destin_region[i] for i in range(8)]) + 1
        if num_curr_region >4:
            y = 0.60+0.4*(8 - num_curr_region)
            test = 8-num_curr_region
        else:
            y = 0.60+0.4*(4 - num_curr_region)
            test = 4-num_curr_region

        print('MoveToRightSide: move right, distance to corner: ', test)
        if num_destin_region > 4:
            curr_region = [0,0,0,0,1,0,0,0]
        else:
            curr_region = [1,0,0,0,0,0,0,0]

        print('temp destination region: ', curr_region)
        return [('MoveBaseRel', [0, -y, 0, V, OMEGA]),   # move to RHS
                ('MoveBaseRel', [2, 0, 0, V, OMEGA]),  # move to other side of the table
                ('MoveBaseRel', [0, 0, 3.14, V, OMEGA]),
                ('MoveBaseRel', [0, -0.750, 0, V, OMEGA]),
                ('MoveAlongSide', curr_region + destin_region),
                (None, None)][cnt]


class MoveToLeftSide(hierarchical.Skill):
    model_name = 'table'
    # model_name = 't_log_poly2'
    # model_name = 'log_poly3'
    arg_in_len = 16  # t-dep logpoly2
    sub_skill_names = ['MoveBaseRel', 'MoveAlongSide']
    ret_out_len = 0
    max_cnt = 6
    sub_arg_accuracy = [1e-2 for i in range(16)]

    # try different model with construct training set.

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        curr_region = arg[0:8]
        destin_region = arg[8:16]
        num_curr_region = sum([i*curr_region[i] for i in range(8)]) + 1
        num_destin_region = sum([i*destin_region[i] for i in range(8)]) + 1
        if num_curr_region > 4:
            y = 0.60 + 0.4 * (num_curr_region - 5)
            test=num_curr_region-5
        else:
            y = 0.60 + 0.4 * (num_curr_region-1)
            test=num_curr_region-1
        print('MoveToLeftSide: move left, distance to corner: ', test)

        if num_destin_region > 4:
            curr_region = [0,0,0,0,0,0,0,1]
        else:
            curr_region = [0,0,0,1,0,0,0,0]
        print('temp destination region: ', curr_region)

        return [('MoveBaseRel', [0, y, 0, V, OMEGA]),   # move to LHS
                ('MoveBaseRel', [2, 0, 0, V, OMEGA]),  # move to other side of the table
                ('MoveBaseRel', [0, 0, 3.14, V, OMEGA]),
                ('MoveBaseRel', [0, 0.750, 0, V, OMEGA]),
                ('MoveAlongSide', curr_region + destin_region),
                (None, None)][cnt]


"""Inspect Table Subskills"""


class GoToTable(hierarchical.Skill):
    # no subskill, poly_linear
    arg_in_len = 17
    sub_skill_names = ['MoveBaseAbs']
    ret_out_len = 17

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: motion_cnt, region_list
        ret_val: after MoveBaseAbs: motion_cnt, region_list
        """
        motion_cnt = arg[0]

        # rewrite this into a linear equation, can be fitted with linear regression
        destin_x = 0
        destin_y = -0.5*(motion_cnt-1)
        # if motion_cnt == 1:
        #     destin_x = 0
        #     destin_y = -0
        # elif motion_cnt == 2:
        #     destin_x = 0
        #     destin_y = -0.5
        if cnt == 0:
            return 'MoveBaseAbs', [0, destin_y, 0, V, OMEGA]
        else:
            # return out value: [motion_cnt, region_list]
            return None, arg


class LookAtTable(hierarchical.Skill):
    # no subskill, poly_linear
    arg_in_len = 17
    sub_skill_names = ['ScanTable']
    ret_out_len = 17

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [motion_cnt, region_list]
        ret_val: motion_cnt, region_list
        """
        motion_cnt = arg[0]
        # region_list = [current_position[0:8], region1_cup_no, region2_cup_no, ..., region8_cup_no]
        region_list = arg[1:17]
        if cnt == 0:
            if motion_cnt == 1:
                return 'ScanTable', region_list
            elif motion_cnt == 2:
                return 'ScanTable', region_list
        else:
            return None, [motion_cnt] + ret_val[0:16]


"""Fill Cup and its Subskills"""


class FillCup(hierarchical.Skill):
    # T-dep logpoly2
    model_name = 't_log_lin'
    arg_in_len = 17
    sub_skill_names = ['GoToCup', 'PourWater']
    ret_out_len = 16
    max_cnt = 4

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: region_list + destin_region + [water_level]
        ret_val:
        """

        region_list = arg[0:16]
        water_level = arg[16]
        if DEBUG:
            return None, region_list
        if cnt == 0:
            # print(arg+[cnt])
            return 'GoToCup', [cnt]
        elif cnt == 1:
            if WaterEnv.cup_regions[ret_val[0]] is None:
                return None, ret_val
            else:
                return 'GoToCup', [cnt]
        elif cnt == 2:
            # ret_val is cup_region, cup_no
            return 'PourWater', [water_level]
        else:
            return None, region_list


class GoToCup(hierarchical.Skill):
    # t-dep log-poly
    arg_in_len = 1
    max_cnt = 4
    sub_skill_names = ['MoveHead', 'LocateObject', 'MoveToLocation']
    ret_out_len = 1

    # return value: Boolean move_succeed

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [region_list, destin_region, motion_cnt]
        region_list = [[cup_region, cup_number],...]
        ret_val: after MoveToLocation: cup_region, cup_no, region_list
        """

        motion_cnt = arg[0]
        # select next serve region

        # if cnt == 0:
        #     # Here cup_no serve as an indicator of moving to LocateObject destination.
        #     return 'MoveToLocation', [destin_region, -1, 0, 0, region_list]
        if cnt == 0:
            return 'MoveHead', [-math.pi / 4., 0.]
        elif cnt == 1:
            return 'LocateObject', [motion_cnt]
        elif cnt == 2:
            [found, cup_pixel_x, cup_pixel_y, theta] = ret_val
            # select next serve cup
            if found==False:
                return None, [found]  # terminate
            else:
                return 'MoveToLocation', [cup_pixel_x, cup_pixel_y, theta]  # found, move to location
        else:
            return None, [True]


class PourWater(hierarchical.Skill):
    arg_in_len = 1
    sub_skill_names = ['MoveArm', 'ArmPourWater']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: cup_region, cup_no, region_list, water_level
        ret_val: cup_region, cup_no, region_list
        """
        water_level= arg[0]
        # region_list = arg[2:18]
        return [
            # ('MoveGripper', [1.57]),
            # ('MoveArm', [0, -0, -1.57, -1.57, 1.02]),
            # ('MoveBaseRel', [0, -0.05, 0, V, OMEGA]),
            # ('MoveGripper', [1.57]),
            # ('MoveArm', [0, -0, -1.57, -1.57, 0]),
            # ('MoveArm', [0.2, -1.2, -1.57, -1.57, 2.3]),
            # ('MoveArm', [0.2, -1.2, -1.57, -1.57, 1.7]),
            # ('MoveArm', [0.2, -1.2, -1.57, -1.57, 2.3]),
            # ('MoveArm', [0, -0, -1.57, -1.57, 1.1]),
            ('ArmPourWater', [water_level]),
            ('MoveArm', [0, -0, -1.57, -1.57, 1.02]),
            (None, None)
        ][cnt]


"""Ensure Water Amount and its Subskills"""


class EnsureWaterAmount(hierarchical.Skill):
    #
    arg_in_len = 8
    sub_skill_names = ['CheckWaterAmount', 'RefillBottle']
    ret_out_len = 2

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [curr_region]
        ret_val: [Water_level, IsRefilled]
        """
        curr_region = arg
        if cnt == 0:
            return 'CheckWaterAmount', None
        elif cnt == 1:
            [Water_Level] = ret_val
            if Water_Level == 0:
                return 'RefillBottle', [Water_Level]+curr_region
            else:
                return None, [Water_Level, False]
        else:
            # ret_val from RefillBottle is [Water_level, True]
            return None, ret_val


class CheckWaterAmount(hierarchical.Skill):
    # log_linear
    arg_in_len = 0
    # TODO: write a action for checking water level in bottle.
    #  It can be a read from a water pressure sensor. Now just put MoveToLocation
    sub_skill_names = ['CheckWaterAmountAction']
    ret_out_len = 1

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: None
        ret_val: Water_Level(Quantized water level)
        """
        return [('CheckWaterAmountAction', None),
                (None, obs)][cnt]


class RefillBottle(hierarchical.Skill):
    # t-dep logpoly2
    arg_in_len = 9
    sub_skill_names = ['GoToRefillStation', 'RefillWater', 'MoveBaseAbs', 'MoveNextRegion']
    ret_out_len = 2

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: Water_Level, current region
        ret_val: Water_Level
        """
        # move to region 0
        water_level = arg[0]
        curr_region = arg[1:9]
        if cnt == 0:
            return 'MoveNextRegion', curr_region + [0 for i in range(8)] + [5, 0]
        if cnt == 1:
            # return 'MoveBaseAbs', [30]
            return 'MoveBaseAbs', STATION_POSITION + [-1.57, V, OMEGA]
        elif cnt == 2:
            return 'GoToRefillStation', None
        elif cnt == 3:
            if WaterEnv.station[ret_val[0]] is None:
                return None, ret_val
            else:
                return 'GoToRefillStation', None
        elif cnt == 4:
            return 'RefillWater', None
        else:
            return None, [5, True]


class GoToRefillStation(hierarchical.Skill):
    # t-dep logpoly2
    arg_in_len = 0
    sub_skill_names = ['MoveToLocation', 'LocateStation', 'MoveHead']
    ret_out_len = 1

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: None
        ret_val:
        """
        if cnt == 0:
            return 'MoveHead', [-math.pi / 4., 0.]
        elif cnt == 1:
            return 'LocateStation', [cnt]
        elif cnt == 2:
            [found_station, sta_pixel_x, sta_pixel_y, theta] = ret_val
            if found_station is None:
                return None, [WaterEnv.station[found_station]]  # terminate
            else:
                # just give it an empty region_list. If station found, found_station = 10
                return 'MoveToLocation', [sta_pixel_x, sta_pixel_y, theta]
        else:
            return None, [1]


class RefillWater(hierarchical.Skill):
    # table
    arg_in_len = 0
    sub_skill_names = ['PlaceBottle', 'Delay', 'TakeBottle']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg:
        ret_val:
        """
        if cnt == 0:
            return 'PlaceBottle', None
        elif cnt == 1:
            return 'Delay', [2]
        elif cnt == 2:
            return 'TakeBottle', None
        else:
            return None, None


"""Lower Level Skills"""


class MoveToLocation(hierarchical.Skill):
    arg_in_len = 3
    sub_skill_names = ['MoveBaseRel']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: [cup_region, cup_no, cup_pixel_x, cup_pixel_y]
        ret_val: [cup_region, cup_no, region_list]
        """
        if DEBUG:
            return None, None
        [x, y, theta] = arg
        if cnt == 0:
            if PRETRAINED:
                # translate pixel location to robot location
                # [x, y, theta] = MoveToLocation.teacher_model.predict(
                # [utils.one_hot({1: 1, 2: 3}[obj_class], len(DishesEnv.obj_classes) + 1)
                # + [obj_pixel_x, obj_pixel_y]])[0]

                # modification: select a object on the table
                # f = open('current_region', 'r')
                # cup_region = int(f.readline())
                # f.close()
                # if cup_pixel_x == 100 and cup_pixel_y == 100:
                #     cup_region = 10
                # gzpos = gazebo_getpos.Gz_getPos()
                # pos_dict = gzpos.pos_lib
                # input = open('target_cup.pkl')
                # item_name = pickle.load(input)
                # input.close()
                # armlength = 0.55
                # y_adjust = 0.1
                # print(pos_dict[item_name])
                # print(pos_dict["hsrb"])
                # # only works when hsrb face the direction of +y axi.
                # if (0<cup_region<5):    # TODO change to 0-7
                #     # In this case, x for hsrb is +y in gazebo, y for hsrb is -x in gazebo
                #     theta_0 = 1.55
                #     theta = theta_0 - pos_dict['hsrb'].z
                #     if abs(theta) < 0.01:
                #         theta = 0
                #     y = -(pos_dict[item_name].x - pos_dict["hsrb"].x) + y_adjust
                #     x = pos_dict[item_name].y - armlength - pos_dict["hsrb"].y
                # else:
                #     if cup_region == 10: item_name = 'station'
                #     theta_0 = -1.55
                #     theta = theta_0 - pos_dict['hsrb'].z
                #     if abs(theta) < 0.01:
                #         theta = 0
                #     y = (pos_dict[item_name].x - pos_dict["hsrb"].x) + 0.075
                #     x = -(pos_dict[item_name].y + armlength - pos_dict["hsrb"].y)
                return 'MoveBaseRel', [x, y, theta, V, OMEGA]
                # take pixel location and move to it. constant: v: velocity, omega:angular v
            else:
                return 'Record_MoveBaseRel', None  # this will be demonstrated by teleoperation
        else:
            return None, None


class PlaceBottle(hierarchical.Skill):
    arg_in_len = 0
    sub_skill_names = ['MoveArm']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: None
        ret_val: None
        """
        print('PlaceBottle: Bottle placed on the station.')
        return [['MoveArm', [0, -0.4, -1.57, -1.57, 1.42]],
                [None, None]][cnt]


class TakeBottle(hierarchical.Skill):
    arg_in_len = 0
    sub_skill_names = ['MoveArm']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: None
        ret_val: None
        """
        return [['MoveArm', [0, -0, -1.57, -1.57, 1.02]],
                [None, None]][cnt]


class GrabPitcher(hierarchical.Skill):
    arg_in_len = 0
    sub_skill_names = ['MoveArm', 'MoveGripper', 'MoveBaseRel']
    ret_out_len = 0

    @staticmethod
    def step(arg, cnt, ret_name, ret_val, obs):
        """
        arg: cup_region, cup_no, region_list, water_level
        ret_val: cup_region, cup_no, region_list
        """
        [cup_region, cup_no, region_list, water_level] = arg
        return [
            ('MoveArm', [0, -0.7, -1.57, -1.57, 0.7]),
            ('MoveBaseRel', [0, -0.05, 0, V, OMEGA]),
            ('MoveGripper', [1.57]),
            ('MoveArm', [0, -0.6, -1.57, -1.57, 0.6]),
            (None, [cup_region, cup_no] + region_list)
        ][cnt]


class WaterAgent(hierarchical.HierarchicalAgent):
    root_skill_name = 'ServeTables'
    # root_skill_name = 'FillCups'
    # root_skill_name = 'MoveToLocation'
    skills = [
        ServeTables,
        ServeTable,
        GoStandBy,
        InspectTable,
        FillCups,#
        FillCupsWrapper,#
        SelectNextRegionWrapper,
        SelectNextRegion,#
        ShiftRegion,
        CheckRegion,
        MoveNextRegion,
        MoveAroundTable,#
        MoveAlongSide,
        MoveToRightSide,
        MoveToLeftSide,
        GoToTable,
        LookAtTable,
        FillCup,#
        GoToCup,
        PourWater,
        EnsureWaterAmount,#
        CheckWaterAmount,#
        RefillBottle,
        GoToRefillStation,
        RefillWater,
        MoveToLocation,#
        PlaceBottle,
        TakeBottle,
        # GrabPitcher
    ]


    actions = WaterEnv.actions
    default_model_name = 'log_poly2'


    def __init__(self, config):
        super(WaterAgent, self).__init__(config)
