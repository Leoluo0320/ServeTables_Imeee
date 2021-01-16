import subprocess
import rospy
import pickle


pkl_file = open('/home/lme/HIL-MT/watered_cup.pkl', 'wb')
pickle.dump([],pkl_file)
pkl_file.close()

a = subprocess.check_output('ps ax | grep /usr/bin/python\ /opt/ros/kinetic/bin/roslaunch\ watertest.launch', shell=True)
pinfo = a.split('\n')
for info in pinfo:
    if "grep" in info:
        continue
    else:
        oput = info.split(' ')
        i=0
        while oput[i] == '':
            i+=1
        pid = oput[i]
        print(oput, pid)
        break
subprocess.check_output('kill -2 {}'.format(str(pid)), shell=True)
rospy.sleep(30)
subprocess.check_output('play -nq -t alsa synth 1 sine 440', shell=True)

# a = " 1312 pts/19   S+     0:00 grep --color=auto /usr/bin/python /opt/ros/kinetic/bin/roslaunch watertest.launch\n22132 pts/18   Sl+    0:21 /usr/bin/python /opt/ros/kinetic/bin/roslaunch watertest.launch"
