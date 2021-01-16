import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
import random
import pickle

# generate random cups on table (table area at  x:(0, 1.6), y:(1.6, 2.4))
# cups is generated at area (x:(0.1,1.5), y:(1.6, 1.8)), (x:(0.1,1.5), y:(2.1, 2.3))

foo = open('water_level', 'w')
foo.write(str(random.randint(1,5)))
foo.close()
foo = open('watered_cup.pkl', 'wb')
pickle.dump([], foo)
foo.close()
rospy.sleep(10)
cup_num = random.randint(3,6)
rospy.init_node('insert_object', log_level=rospy.INFO)
x = [0 for temp in range(0,cup_num)]
y = [0 for temp in range(0,cup_num)]
for i in range(0,cup_num):
    yside = random.randint(0,1)
    while(True):
        if yside:
            y[i] = 1.65+0.1*random.random()
        else:
            y[i] = 2.25+0.1*random.random()
        x[i] = 0.1+1.3*random.random()
        valid = True
        for r in range(0,i):
            if y[r]-0.1<y[i]<y[r]+0.1 and x[r]-0.2<x[i]<x[r]+0.2:
                valid = False
                break
        if valid:
            break
        else:
            continue

    initial_pose = Pose()
    initial_pose.position.x = x[i]
    initial_pose.position.y = y[i]
    initial_pose.position.z = 0.48

    f = open('/home/lme/model_editor_models/blue_cup/model.sdf', 'r')
    sdff = f.read()

    rospy.wait_for_service('gazebo/spawn_sdf_model')
    spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
    spawn_model_prox("cup_{}".format(str(i)), sdff, "robotos_name_space", initial_pose, "world")
x = 0
