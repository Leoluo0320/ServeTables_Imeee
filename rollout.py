import argparse
import os
import pickle
import time
from datetime import datetime
import rospy
import random
import agents
import envs
from utils import DictTree


def rollout(config):
    # modification: pass ClearTable parameter for easy debugging
    # If evaluate settable, it returns DishesEnv in envs/dishes.py
    foo = open('water_level', 'w')
    foo.write(str(random.randint(1, 5)))
    foo.close()
    foo = open('watered_cup.pkl', 'wb')
    pickle.dump([], foo)
    foo.close()
    rospy.sleep(10)
    rospy.sleep(5)
    config.domain = 'water'
    config.task = 'ServeWater'
    config.model = '/home/lme/HIL-MT/model/water'
    config.data = 'eval'
    config.teacher = False
    env = envs.catalog(config.domain)   # For ClearTable, env is dishes.DishesEnv
    # go to agents __init__.py. return a class which contain all skills about Settable or Pyramid.
    # If evaluate settable, it returns DishesAgent in agents/dishes.py
    agent = agents.catalog(DictTree(domain_name=config.domain, task_name=config.task, teacher=config.teacher, rollable=True, model_dirname=config.model))
    # init_arg = env.reset(config.task)
    init_arg = [0]
    # init_arg = [1,0,0,0,0,0,0,0, 1,0,1,1,0,0,0,0]

    agent.reset(init_arg)
    trace = agent.rollout(env)
    try:
        os.makedirs("{}/{}".format(config.data, config.domain))
    except OSError:
        pass
    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    # pickle.dump(trace, open("{}/{}/{}.{}.pkl".format(config.data, config.domain, config.task, time_stamp), 'wb'), protocol=2)
    pickle.dump(trace, open("{}/{}/{}.{}.pkl".format(config.data, config.domain, config.task, time_stamp), 'wb'), protocol=2)
    print("=== trace saved ===")
    time.sleep(1)
    # raw_input("Press Enter to continue...")

def DataCollectForRegionList(config):
    # modification: pass ClearTable parameter for easy debugging
    # If evaluate settable, it returns DishesEnv in envs/dishes.py
    rospy.sleep(5)
    config.domain = 'water'
    config.task = 'FillCups'
    config.model = '/home/lme/HIL-MT/model/water'
    config.data = 'eval'
    config.teacher = False
    for i in range(1, 9):
        for j in range(1, 256):
            water_level = random.randint(0,5)
            foo = open('water_level', 'w')
            foo.write(str(1))
            foo.close()
            print 'water level is: ', water_level

            temp = j
            emcup_per_region=[]
            curr_region = [0 for x in range(i - 1)] + [1] + [0 for x in range(8 - i)]
            emcup_per_region.append((temp >> 7) % 2)
            emcup_per_region.append((temp >> 6) % 2)
            emcup_per_region.append((temp >> 5) % 2)
            emcup_per_region.append((temp >> 4) % 2)
            emcup_per_region.append((temp >> 3) % 2)
            emcup_per_region.append((temp >> 2) % 2)
            emcup_per_region.append((temp >> 1) % 2)
            emcup_per_region.append((temp >> 0) % 2)

            env = envs.catalog(config.domain)   # For ClearTable, env is dishes.DishesEnv
            # go to agents __init__.py. return a class which contain all skills about Settable or Pyramid.
            # If evaluate settable, it returns DishesAgent in agents/dishes.py
            agent = agents.catalog(DictTree(domain_name=config.domain, task_name=config.task, teacher=config.teacher, rollable=True, model_dirname=config.model))
            # init_arg = env.reset(config.task)
            # init_arg = curr_region + emcup_per_region
            init_arg = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
            # init_arg = [1]
            agent.reset(init_arg)
            trace = agent.rollout(env)
            try:
                os.makedirs("{}/{}".format(config.data, config.domain))
            except OSError:
                pass
            time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            # pickle.dump(trace, open("{}/{}/{}.{}.pkl".format(config.data, config.domain, config.task, time_stamp), 'wb'), protocol=2)
            f = open("{}/{}/{}.{}.pkl".format(config.data, config.domain, config.task, time_stamp), 'wb')
            pickle.dump(trace, f, protocol=2)
            time.sleep(1)
            f.close()
            print("=== trace saved ===")
    # raw_input("Press Enter to continue...")

def DataCollectForTable(config):
    # modification: pass ClearTable parameter for easy debugging
    # If evaluate settable, it returns DishesEnv in envs/dishes.py
    rospy.sleep(5)
    config.domain = 'water'
    config.task = 'FillCups'
    config.model = '/home/lme/HIL-MT/model/water'
    config.data = 'eval'
    config.teacher = False
    for i in range(1, 9):
        for j in range(0, 9):
            water_level = random.randint(0,5)
            foo = open('water_level', 'w')
            foo.write(str(water_level))
            foo.close()
            print 'water level is: ', water_level

            temp = j
            curr_region = [0 for x in range(i - 1)] + [1] + [0 for x in range(8 - i)]
            destin_region = [0 for x in range(j - 1)] + [1] + [0 for x in range(8 - j)]
            if j == 0:
                destin_region = [0 for _ in range(8)]

            env = envs.catalog(config.domain)   # For ClearTable, env is dishes.DishesEnv
            # go to agents __init__.py. return a class which contain all skills about Settable or Pyramid.
            # If evaluate settable, it returns DishesAgent in agents/dishes.py
            agent = agents.catalog(DictTree(domain_name=config.domain, task_name=config.task, teacher=config.teacher, rollable=True, model_dirname=config.model))
            # init_arg = env.reset(config.task)
            init_arg = curr_region+destin_region
            # init_arg = [1]
            agent.reset(init_arg)
            trace = agent.rollout(env)
            try:
                os.makedirs("{}/{}".format(config.data, config.domain))
            except OSError:
                pass
            time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            # pickle.dump(trace, open("{}/{}/{}.{}.pkl".format(config.data, config.domain, config.task, time_stamp), 'wb'), protocol=2)
            f = open("{}/{}/{}.{}.pkl".format(config.data, config.domain, config.task, time_stamp), 'wb')
            pickle.dump(trace, f, protocol=2)
            time.sleep(1)
            f.close()
            print("=== trace saved ===")
    # raw_input("Press Enter to continue...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--domain', required=True)
    # parser.add_argument('--task', required=True)
    # parser.add_argument('--model')
    # parser.add_argument('--data')
    # parser.add_argument('--teacher', action='store_true')
    args = parser.parse_args()
    rollout(args)
    # DataCollectForTable(args)
    # DataCollectForRegionList(args)

    # rollout(domain = 'dishes', task = 'ClearTable', model = 'model', data = 'eval', teacher = False)