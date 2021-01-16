import math

import numpy as np

import vision
from utils import DictTree

try:
    import ar_markers
    import control_msgs.msg as ctrl_msg
    import cv2
    import cv_bridge
    import rospy
    import sensor_msgs.msg as sens_msg
    import tmc_control_msgs.msg as tmc_msg
    import trajectory_msgs.msg as traj_msg
    from agents import gazebo_getpos # modification: LocateObject will use Gz_getPos
    import os, pickle, time
except ImportError, e:
    print(e)

import env

V = 0.07
OMEGA = math.pi / 5.
ARM_STAND_BY = [0, -0.4, -1.57, -1.57, 1.42]
DEBUG = 0



class MoveArm(env.Action):
    arg_in_len = 5
    ret_out_len = 0
    @staticmethod
    def apply(hsr, arg):
        if DEBUG:
            pass
        else:
            # joint from low to high. arm_lift: 1st joint, lift arm. arm_flex: 2nd joint should be negative in rad,
            # arm_roll: 3rd joint, rotate. wrist_flex:4th joint wrist_roll:5th joint rotate.
            arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll = arg[:5]
            point = traj_msg.JointTrajectoryPoint()  # initialize an object of class JointTrajectoryPoint
            point.positions = [arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll]
            point.velocities = [0] * 5  # velocity
            point.time_from_start = rospy.Time(0.5) # complete in 0.5 sec
            traj = ctrl_msg.FollowJointTrajectoryActionGoal()    # initialize an object of class FollowJointTrajectoryActionGoal
            traj.goal.trajectory.joint_names = ['arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint', 'wrist_flex_joint',
                                                'wrist_roll_joint']
            traj.goal.trajectory.points = [point]
            hsr.publishers['arm'].publish(traj)  # publisher node arm publish generated trajectory
            rospy.sleep(1.)
            # recursively checking error
            while True:
                error = hsr.subscribers['arm_pose'].callback.data['error']
                if all(abs(x) <= 1e-2 for x in error):  # 1e-2: scientific notation
                    break
                rospy.sleep(0.1)    # 1000Hz?


class ArmPourWater(env.Action):
    arg_in_len = 5
    ret_out_len = 0

    @staticmethod
    def apply(hsr, arg):
        if DEBUG:
            pass
        else:
            water_level = int(round(arg[0]))
            max_angle = 1   #0.785
            a = 0.5
            r = 0.23
            alpha_0 = -ARM_STAND_BY[1]
            trajctory_pts = []
            for i in range(0, (int)(max_angle//0.1)):
                phi = max_angle/(max_angle//0.1)*i
                alpha = np.arcsin((a*np.sin(alpha_0)+r-r*np.cos(phi))/a)
                delta_d = a*np.cos(alpha_0)+r*np.sin(phi)-a*np.cos(alpha)
                if i>5:
                    alpha=alpha*1.2
                pt = [ARM_STAND_BY[0]+delta_d, -alpha] + ARM_STAND_BY[2:4] + [ARM_STAND_BY[4]-phi]
                trajctory_pts.append(pt)

            # phi = 0.785
            # alpha = np.arcsin((a * np.sin(alpha_0) + r - r * np.cos(phi)) / a)
            # delta_d = a * np.cos(alpha_0) + r * np.sin(phi) - a * np.cos(alpha)
            # pt = [ARM_STAND_BY[0] + delta_d, -alpha] + ARM_STAND_BY[2:4] + [ARM_STAND_BY[4] - phi]
            # trajctory_pts.append(pt)

            for i in range(0,(len(trajctory_pts)-2)*(6-water_level)/5+2):
                # joint from low to high. arm_lift: 1st joint, lift arm. arm_flex: 2nd joint should be negative in rad,
                # arm_roll: 3rd joint, rotate. wrist_flex:4th joint wrist_roll:5th joint rotate.
                arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll = trajctory_pts[i]
                point = traj_msg.JointTrajectoryPoint()  # initialize an object of class JointTrajectoryPoint
                point.positions = [arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll]   #multiply by a coefficient
                if i > 0:
                    point.velocities = [(arm_lift-trajctory_pts[i-1][0])*2,
                                        (arm_flex-trajctory_pts[i-1][1])*2,
                                        0,
                                        0,
                                        (wrist_roll-trajctory_pts[i-1][4])*2]  # velocity
                else:
                    point.velocities=[0]*5
                point.time_from_start = rospy.Time(0.5) # complete in 0.5 sec
                traj = ctrl_msg.FollowJointTrajectoryActionGoal()    # initialize an object of class FollowJointTrajectoryActionGoal
                traj.goal.trajectory.joint_names = ['arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint', 'wrist_flex_joint',
                                                    'wrist_roll_joint']
                traj.goal.trajectory.points = [point]
                hsr.publishers['arm'].publish(traj)  # publisher node arm publish generated trajectory
                rospy.sleep(0.5)
                # recursively checking error
                while True:
                    error = hsr.subscribers['arm_pose'].callback.data['error']
                    if all(abs(x) <= 1e-2 for x in error):  # 1e-2: scientific notation
                        break
                    rospy.sleep(0.1)    # 1000Hz?
            foo = open('water_level', 'w')
            foo.write(str(water_level-1))
            foo.close()
            rospy.sleep(0.5)


class MoveBaseAbs(env.Action):
    arg_in_len = 5
    ret_out_len = 0

    @staticmethod
    def apply(hsr, arg):
        if DEBUG:
            pass
        else:
            # hsr here is an instance of hsr.HSREnv, argument is a list of 5 numbers
            x, y, theta, v, omega = arg[:5]
            x0, y0, theta0 = hsr.subscribers['base_pose'].callback.data['actual']
            dx = ((x - x0) ** 2 + (y - y0) ** 2) ** .5  # distance in length
            dtheta = abs(theta - theta0)    # distance in angle
            t = max(dx / v, dtheta / omega)
            acceleration = 0.05
            point = traj_msg.JointTrajectoryPoint()
            point.positions = [x, y, theta]
            point.accelerations = [acceleration] * 3
            point.effort = [1]
            point.time_from_start = rospy.Time(t)
            traj = ctrl_msg.FollowJointTrajectoryActionGoal()
            traj.goal.trajectory.joint_names = ['odom_x', 'odom_y', 'odom_t']
            traj.goal.trajectory.points = [point]
            hsr.publishers['base'].publish(traj)
            rospy.sleep(1.)
            while True:
                error = hsr.subscribers['base_pose'].callback.data['error']
                if all(abs(x) <= 1e-6 for x in error):
                    break
                rospy.sleep(0.1)


class MoveBaseRel(env.Action):
    arg_in_len = 5
    ret_out_len = 0

    @staticmethod
    def apply(hsr, arg):
        if DEBUG:
            pass
        else:
            x, y, theta, v, omega = arg[:5]
            x0, y0, theta0 = hsr.subscribers['base_pose'].callback.data['actual']
            x, y, theta = (
                x * math.cos(theta0) - y * math.sin(theta0) + x0,
                x * math.sin(theta0) + y * math.cos(theta0) + y0,
                theta + theta0)
            MoveBaseAbs.apply(hsr, [x, y, theta, v, omega])


class MoveGripper(env.Action):
    arg_in_len = 1
    ret_out_len = 0

    @staticmethod
    def apply(hsr, arg):
        [close] = arg[:1]
        if close:
            rospy.sleep(1.)
        grip = tmc_msg.GripperApplyEffortActionGoal()
        grip.goal.effort = -0.1 if close > 0.5 else 0.1
        hsr.publishers['grip'].publish(grip)
        rospy.sleep(1.)


class MoveHead(env.Action):
    arg_in_len = 2
    ret_out_len = 0

    @staticmethod
    def apply(hsr, arg):
        if DEBUG:
            pass
        else:
            tilt, pan = arg[:2]
            point = traj_msg.JointTrajectoryPoint()
            point.positions = [tilt, pan]
            point.velocities = [0] * 2
            point.time_from_start = rospy.Time(0.5)
            traj = ctrl_msg.FollowJointTrajectoryActionGoal()
            traj.goal.trajectory.joint_names = ['head_tilt_joint', 'head_pan_joint']
            traj.goal.trajectory.points = [point]
            hsr.publishers['head'].publish(traj)
            rospy.sleep(2.)


class LocateObject(env.Action):
    arg_in_len = 1
    ret_out_len = 4

    @staticmethod
    def apply(hsr, arg):
        # motion_cnt, obj_class, obj_color = arg[:3]
        # img = hsr.subscribers['head_cam'].callback.data
        # objs = hsr.vision.get_objs(img, 0.1)    # pyyolo
        # objs = list(sorted((obj for obj in objs if obj['class'] in hsr.obj_classes_filter), key=lambda o: o['bottom'],
        #                    reverse=True))
        # detected_obj_class = hsr.obj_classify(
        #     [[motion_cnt, obj['left'], obj['top'], obj['right'], obj['bottom']] for obj in objs])
        # for i, obj in enumerate(objs):
        #     obj['class_idx'] = detected_obj_class[i]
        #     obj['mean_color'] = img[obj['top']:obj['bottom'], obj['left']:obj['right'], :].mean(0).mean(0) - img.mean(
        #         0).mean(0)
        #     obj['color_idx'] = 1 + np.argmax(obj['mean_color'])
        # for i, obj in enumerate(objs):
        #     box_color = (255, 0, 0)
        #     cv2.rectangle(img, (obj['left'], obj['top']), (obj['right'], obj['bottom']), box_color, 4)
        #     cv2.putText(
        #         img, "{} {}".format(hsr.obj_colors[obj['color_idx']], hsr.obj_classes[obj['class_idx']]),
        #         (obj['left'], obj['top']), cv2.FONT_HERSHEY_PLAIN, 2, box_color, 1, 8)
        # cv2.imshow('head', img)
        # cv2.waitKey(3)
        # if abs(obj_class - hsr.obj_classes.index(None)) > 0.5:
        #     objs = [obj for obj in objs if abs(obj['class_idx'] - obj_class) < 0.5]
        # if abs(obj_color - hsr.obj_colors.index(None)) > 0.5:
        #     objs = [obj for obj in objs if abs(obj['color_idx'] - obj_color) < 0.5]
        # if len(objs) > 0:
        #     obj = objs[0]
        #     box_color = (0, 255, 0)
        #     cv2.rectangle(img, (obj['left'], obj['top']), (obj['right'], obj['bottom']), box_color, 4)
        #     cv2.putText(
        #         img, "{} {}".format(hsr.obj_colors[obj['color_idx']], hsr.obj_classes[obj['class_idx']]),
        #         (obj['left'], obj['top']), cv2.FONT_HERSHEY_PLAIN, 2, box_color, 1, 8)
        #     return [obj['class_idx'], obj['color_idx'], (obj['left'] + obj['right']) / 2,
        #             (obj['top'] + obj['bottom']) / 2]
        # else:
        #     return [0, 0, 0, 0]
        if DEBUG:
            return [True, 0,0,0]
        motion_cnt = int(round(arg[0]))
        # region_list = arg[1:17]

        # initialize the getpos class
        gzpos = gazebo_getpos.Gz_getPos()
        pos_dict = gzpos.pos_lib

        if motion_cnt == 0:
            # calculate robot's current region
            curr_region = (int)(pos_dict['hsrb'].x // 0.375 + 1)
            if curr_region > 4:
                curr_region = 4
            elif curr_region < 1:
                curr_region = 1
            if pos_dict['hsrb'].y > 2:
                curr_region = 8-curr_region+1
            print('Locate_Object: hsrb region: ', curr_region, 'x', pos_dict['hsrb'].x, 'y: ', pos_dict['hsrb'].y)

            # construct region that robot could 'see'
            lcurr_region = curr_region-1
            rcurr_region = curr_region+1
            if curr_region == 1 or curr_region == 5:
                lcurr_region = curr_region
            if curr_region == 4 or curr_region == 8:
                rcurr_region = curr_region
            curr_list = []
            lcurr_list = []
            rcurr_list = []

            cup_selected = False
            cup_region = None
            if os.path.exists('watered_cup.pkl'):
                input = open('watered_cup.pkl')
                watered_cups = pickle.load(input)
            else:
                watered_cups = []
            for mname in pos_dict.allkeys():
                model_name = mname[0]
                # object in gazebo that isn't cups
                if model_name == 'hsrb' or model_name == 'ground_plane' or model_name == 'IKEAtable' or model_name == 'Pitcher' or model_name == 'station' or model_name in watered_cups:
                    continue

                # calculate cup region
                # Table is at (0, 1.6) (1.6, 2.4)
                cup_region = (int)(pos_dict[mname].x//0.375+1)
                if (pos_dict[mname].y-1.6)//0.4 == 1:
                    cup_region = 8-cup_region+1

                if curr_region == cup_region:
                    curr_list.append(model_name)
                elif lcurr_region == cup_region:
                    lcurr_list.append(model_name)
                elif rcurr_region == cup_region:
                    rcurr_list.append(model_name)
            print(curr_region, lcurr_list, curr_list, rcurr_list)
            if len(lcurr_list) > 0:
                item_name = lcurr_list[0]
                cup_region = lcurr_region
            elif len(curr_list)>0:
                item_name = curr_list[0]
                cup_region = curr_region
            elif len(rcurr_list) > 0:
                item_name = rcurr_list[0]
                cup_region = rcurr_region
            else:
                # terminated
                return [False, 0, 0, 0]
            watered_cups.append(item_name)
            output = open('watered_cup.pkl', 'wb')
            pickle.dump(watered_cups, output)
            output.close()
            output = open('target_cup.pkl', 'wb')
            pickle.dump([item_name, cup_region], output)
            output.close()
        else:
            # motion_cnt == 1
            iput = open('target_cup.pkl')
            [item_name, cup_region] = pickle.load(iput)
            iput.close()

        armlength = 0.55
        y_adjust = 0.1

        # only works when hsrb face the direction of +y axi.
        if (0 < cup_region < 5):  # TODO change to 0-7
            # In this case, x for hsrb is +y in gazebo, y for hsrb is -x in gazebo
            theta_0 = 1.55
            theta = theta_0 - pos_dict['hsrb'].z
            if abs(theta) < 0.01:
                theta = 0
            y = -(pos_dict[item_name].x - pos_dict["hsrb"].x) + y_adjust
            x = pos_dict[item_name].y - armlength - pos_dict["hsrb"].y
        else:
            theta_0 = -1.55
            theta = theta_0 - pos_dict['hsrb'].z
            if abs(theta) < 0.01:
                theta = 0
            y = (pos_dict[item_name].x - pos_dict["hsrb"].x) + 0.075
            x = -(pos_dict[item_name].y + armlength - pos_dict["hsrb"].y)


        ret_val = [True, x, y, theta]
        return ret_val


class ScanTable(env.Action):
    arg_in_len = 16
    ret_out_len = 16

    @staticmethod
    def apply(hsr, arg):
        region_list = arg[0:16]
        if DEBUG:
            # decrease cup_per_region by 1 in curr_region
            region_list[int(round(sum([region_list[idx]*idx for idx in range(0, 8)])+8))] = 0
            return region_list
        gzpos = gazebo_getpos.Gz_getPos()
        pos_dict = gzpos.pos_lib
        # cups that already have water
        f = open('watered_cup.pkl', 'rb')
        watered_cups = pickle.load(f)
        # initialize region list
        for i in range(8, 16):
            region_list[i] = 0
        for mname in pos_dict.allkeys():
            model_name = mname[0]
            # object in gazebo that isn't cups
            if model_name == 'hsrb' or model_name == 'ground_plane' or model_name == 'IKEAtable' or model_name == 'Pitcher' or model_name == 'station' or model_name in watered_cups:
                continue
            # calculate cup region
            # Table is at (0, 1.6) (1.5, 2.4)
            cup_region = (int)(pos_dict[mname].x//0.375+1)
            if (pos_dict[mname].y-1.6)//0.4 == 1:
                cup_region = 8-cup_region+1
            region_list[7+cup_region] = 1
        return region_list

class LocateMarkers(env.Action):    # For pyramid task
    arg_in_len = 0
    ret_out_len = 16

    @staticmethod
    def apply(hsr, arg):
        while True:
            rospy.sleep(5.)
            img = hsr.subscribers['head_cam'].callback.data
            _, img = cv2.threshold(img, 32, 255, cv2.THRESH_BINARY)
            markers = {marker.id: marker for marker in ar_markers.detect_markers(img)}.values()
            for marker in markers:
                marker.draw_contour(img)
            cv2.imshow('head', img)
            cv2.waitKey(3)
            if len(markers) == 2:
                return sort_markers(sum([list(marker.contours.flat) for marker in markers], []))
            else:
                print('found {} markers'.format(len(markers)))
                rospy.sleep(1.)

class LocateStation(env.Action):
    arg_in_len = 1
    ret_out_len = 4

    @staticmethod
    def apply(hsr, arg):
        """
        arg: motion_cnt
        ret_val: bool FindStation, pixel_stationX, pixel_stationY
        """
        if DEBUG:
            return [1, 0,0,0]
        # [motion_cnt] = int(round(arg))
        gzpos = gazebo_getpos.Gz_getPos()
        pos_dict = gzpos.pos_lib
        item_name = 'station'
        theta_0 = -1.55
        theta = theta_0 - pos_dict['hsrb'].z
        armlength = 0.55

        if abs(theta) < 0.01:
            theta = 0
        y = (pos_dict[item_name].x - pos_dict["hsrb"].x) + 0.075
        x = -(pos_dict[item_name].y + armlength - pos_dict["hsrb"].y)
        return [1, x, y, theta]


class CheckWaterAmountAction(env.Action):
    arg_in_len = 0
    ret_out_len = 1

    @staticmethod
    def apply(hsr, arg):
        """
        arg: None
        ret_val: int Water_level
        """
        foo = open('water_level', 'r')
        water_level = int(foo.readline())
        foo.close()
        if water_level == 0:
            foo = open('water_level', 'w')
            foo.write(str(5))
        return [water_level]    # bracket or not?


class Delay(env.Action):
    arg_in_len = 1
    ret_out_len = 0

    @staticmethod
    def apply(hsr, arg):
        """
        arg: second
        ret_val: None
        """
        if DEBUG:
            return
        print('Delay: Time delay for '+str(arg[0])+'s.')
        rospy.sleep(arg[0])

class ReturnRegionList(env.Action):
    arg_in_len = 16
    ret_out_len = 16

    @staticmethod
    def apply(hsr, arg):
        """
        arg: Region list
        ret_val: Region list
        usage: make region list in parent skill to be an obs
        """
        return arg

def sort_markers(markers, sort_between_markers=True):
    assert len(markers) == 16
    markers = [int(x) for x in markers]
    res = []
    for marker in [markers[i:i + 8] for i in range(0, 16, 8)]:
        # sort 4 points from left to right
        points_xy = sorted(marker[i:i + 2] for i in range(0, 8, 2))
        # sort each pair of points from top to bottom, and flatten
        res.append(sum(sum((sorted(points_xy[i:i + 2], key=lambda xy: xy[1]) for i in range(0, 4, 2)), []), []))
    if sort_between_markers:
        return sum(sorted(res), [])
    else:
        return sum(res, [])


class HSREnv(env.Env):
    actions = [
        MoveArm,
        ArmPourWater,
        MoveBaseAbs,
        MoveBaseRel,
        MoveGripper,
        MoveHead,
        LocateObject,
        LocateMarkers,
        ScanTable,
        LocateStation,
        CheckWaterAmountAction,
        Delay,
        ReturnRegionList
    ]
    action_lookup = {action.__name__: action for action in actions}

    def __init__(self):
        if DEBUG == 0:
            rospy.init_node('main', anonymous=True)
            self.subscribers = {
                'head_cam': rospy.Subscriber('/hsrb/head_r_stereo_camera/image_raw', sens_msg.Image, ImageCallback(),
                                             queue_size=1),
                'arm_pose': rospy.Subscriber('/hsrb/arm_trajectory_controller/state',
                                             ctrl_msg.JointTrajectoryControllerState, PoseCallback(), queue_size=1),
                'base_pose': rospy.Subscriber('/hsrb/omni_base_controller/state', ctrl_msg.JointTrajectoryControllerState,
                                              PoseCallback(), queue_size=1),
            }
            self.publishers = {
                'arm': rospy.Publisher('/hsrb/arm_trajectory_controller/follow_joint_trajectory/goal',
                                       ctrl_msg.FollowJointTrajectoryActionGoal, queue_size=1),
                'base': rospy.Publisher('/hsrb/omni_base_controller/follow_joint_trajectory/goal',
                                        ctrl_msg.FollowJointTrajectoryActionGoal, queue_size=1),
                'grip': rospy.Publisher('/hsrb/gripper_controller/grasp/goal', tmc_msg.GripperApplyEffortActionGoal,
                                        queue_size=1),
                'head': rospy.Publisher('/hsrb/head_trajectory_controller/follow_joint_trajectory/goal',
                                        ctrl_msg.FollowJointTrajectoryActionGoal, queue_size=1),
            }
            try:
                if DEBUG:
                    pass
                else:
                    while (any(publisher.get_num_connections() == 0 for publisher in self.publishers.values())
                           or any(subscriber.get_num_connections() == 0 or subscriber.callback.data is None for subscriber in
                                  self.subscribers.values())):
                        if rospy.is_shutdown():
                            raise ValueError()
                        print('Waiting for: {}'.format(
                            [name for name, publisher in self.publishers.items() if publisher.get_num_connections() == 0] +
                            [name for name, subscriber in self.subscribers.items() if
                             subscriber.get_num_connections() == 0 or subscriber.callback.data is None]))
                        rospy.sleep(0.1)
            except KeyboardInterrupt:
                rospy.loginfo(KeyboardInterrupt)
                raise
        # self.vision = vision.Yolo()
        self.obs = []   # modification: change from None to a empty list

    def observe(self):
        return self.obs

    def step(self, act_name, act_arg):
        self.obs = self.action_lookup[act_name].apply(self, act_arg)

    def record(self):
        return [
            self.subscribers['arm_pose'].callback.data,
            self.subscribers['base_pose'].callback.data,
        ]


class DataCallback(object):
    def __init__(self, sleep=0.1):
        self.data = None
        self.sleep = sleep

    def __call__(self, data):
        self.data = data
        rospy.sleep(self.sleep)


class ImageCallback(DataCallback):
    def __init__(self, sleep=0.1):
        super(ImageCallback, self).__init__(sleep)
        self.bridge = cv_bridge.CvBridge()

    def __call__(self, img):
        img = self.bridge.imgmsg_to_cv2(img, 'bgr8')
        img = cv2.resize(img, (920, 690), interpolation=cv2.INTER_CUBIC)
        super(ImageCallback, self).__call__(img)


class PoseCallback(DataCallback):
    def __call__(self, data):
        pose = DictTree(
            actual=data.actual.positions,
            error=data.error.positions,
        )
        super(PoseCallback, self).__call__(pose)
