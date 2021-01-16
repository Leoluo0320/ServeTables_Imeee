source ~/.bashrc
cd ~/HIL-MT
python2.7 cup_generate.py
rosservice call /gazebo/unpause_physics
python2.7 rollout.py
