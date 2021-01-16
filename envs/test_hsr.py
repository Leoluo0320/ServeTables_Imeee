import hsr
import rospy


hsrenvir = hsr.HSREnv()

# mover arm instruction
movearm = hsr.MoveArm()
movearm.apply(hsrenvir, [0, -0, -1.57, -1.57, 1.7])
# movearm.apply(hsrenvir, [0.2, -0.5, -1.57, -1.577, 2.2])


# move head
# moveh = hsr.MoveHead()
# moveh.apply(hsrenvir, [0,0])

# move gripper
# movegrp = hsr.MoveGripper()
# movegrp.apply(hsrenvir, [0])

# move base absolute instruction
# movebaseab = hsr.MoveBaseAbs()
# movebaseab.apply(hsrenvir, [0, 0, 0, 1, 1])

# move base relative instruction
# movebaseab = hsr.MoveBaseRel()
# movebaseab.apply(hsrenvir, [0, 0, -0.1, 1, 1])





