import pybullet as p
import pybullet_data
import math
import time
import os
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

planeId = p.loadURDF('plane.urdf')
pandaId = p.loadURDF('franka_panda/panda.urdf', basePosition=[0,0,0.6],baseOrientation= p.getQuaternionFromEuler([0,0,0]),useFixedBase=True)
tableId = p.loadURDF('table/table.urdf', basePosition=[0.65/2,0,0], baseOrientation=p.getQuaternionFromEuler([0,0, 0]))
rand_objId = p.loadURDF('random_urdfs/000/000.urdf', basePosition = [0.8,0, 0.6]) # some random object
cubeId = p.loadURDF('cube_small.urdf', basePosition = [0.8, -0.1, 1.5])

p.changeDynamics(cubeId, -1, 5) # changing the mass of the cube makes it more stationary when applying fixed constraint later on

num_joints = p.getNumJoints(pandaId)
print(f"Number of joints: {num_joints}")
panda_end_effector_idx = 11 # End effector joint

for t in range(100000):

    target_orn = p.getQuaternionFromEuler([0,-math.pi, math.pi/2]) # downward orientation
    
    cube_orn = p.getQuaternionFromEuler([0,0,0])
    panda_cid = p.createConstraint(pandaId, 11, cubeId, -1, p.JOINT_FIXED, [0,0,0], [0.035, 0, -0.04], childFramePosition = [0,0,0], childFrameOrientation = cube_orn)

    cube_state = p.getBasePositionAndOrientation(cubeId)
    cube_pos = np.array(cube_state[0])
    cube_orn = np.array(cube_state[1])
    rot_matrix = np.array(p.getMatrixFromQuaternion(cube_orn)).reshape(3,3) # 3x3 rotation matrix (right, forward, up by columns)
    forward_vec = rot_matrix.dot(np.array((0, 0, 1)))
    up_vec = rot_matrix.dot(np.array((0, 1, 0)))

    target_position = cube_pos + 0.1 * forward_vec

    view_matrix = p.computeViewMatrix(cube_pos, target_position, up_vec)

    cam_width, cam_height = 960,720
    aspect_ratio = cam_width / cam_height
    fov = 60
    nearVal = 0.01
    farVal = 100

    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, nearVal, farVal)

    img = p.getCameraImage(cam_width, cam_height, view_matrix, proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL)

    if t <= 100: # Panda moves to hover above object 
        target_pos = [0.8, 0, 1.5] 
        gripper = 1 # gripper val is 0 if grasping object, else 1
    elif t > 100 and t <= 200: # Panda moves down
        target_pos = [0.8, 0, 0.63]
        gripper = 1
    elif t > 200 and t <= 300: # Panda grasps object
        target_pos = [0.8, 0, 0.63]
        gripper = 0
    elif t > 300 and t <= 400: # Panda moves up again
        target_pos = [0.8, -0.3, 1.8]
        gripper = 0
    elif t > 400: # Panda drops object 
        gripper = 1
        
    joint_poses = p.calculateInverseKinematics(pandaId, panda_end_effector_idx, target_pos, target_orn) 
    for i in range(7): # 7 DoF
        p.setJointMotorControl2(pandaId, i, controlMode=p.POSITION_CONTROL, targetPosition=joint_poses[i])
    p.setJointMotorControl2(pandaId, 9, controlMode = p.POSITION_CONTROL, targetPosition = gripper * 0.5) # gripper
    p.setJointMotorControl2(pandaId, 10, controlMode = p.POSITION_CONTROL, targetPosition = gripper * 0.5)

    # rgb_array = np.array(rgbPixels, dtype = np.uint8) # 8 bit integers
    # rgb_array = np.reshape(rgb_array, (cam_height, cam_width, 4))
    # depth_array = np.reshape(depthPixels, [cam_height, cam_width])

    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 
    p.stepSimulation()
   # time.sleep(1/240)

p.disconnect()