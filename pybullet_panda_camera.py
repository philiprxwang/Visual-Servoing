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
rand_objId = p.loadURDF('random_urdfs/000/000.urdf', basePosition = [0.5,0, 0.6]) # some random object

p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=50, cameraPitch=-30, cameraTargetPosition=[0.2,0,0.8])

num_joints = p.getNumJoints(pandaId)
print(f"Number of joints: {num_joints}")
panda_end_effector_idx = 11 # End effector joint

for t in range(100000):

     
    target_pos = [0.5, 0, 1.5] 
    target_orn = p.getQuaternionFromEuler([0,-math.pi, math.pi/2]) # downward orientation
    
    joint_poses = p.calculateInverseKinematics(pandaId, panda_end_effector_idx, target_pos, target_orn) 
    for i in range(7): # 7 DoF
        p.setJointMotorControl2(pandaId, i, controlMode=p.POSITION_CONTROL, targetPosition=joint_poses[i])

    end_effector_state = p.getLinkState(pandaId, 11)
    end_effector_pos = np.array(end_effector_state[0])
    end_effector_orn = np.array(end_effector_state[1])
    rot_matrix = np.array(p.getMatrixFromQuaternion(end_effector_orn)).reshape(3,3) # 3x3 rotation matrix (right, forward, up by columns)
    forward_vec = rot_matrix.dot(np.array((0, 0, 1)))
    up_vec = rot_matrix.dot(np.array((0, 1, 0)))

    target_position = end_effector_pos + 0.1 * forward_vec

    view_matrix = p.computeViewMatrix(end_effector_pos, end_effector_pos + 0.1 * forward_vec, up_vec)

    cam_width, cam_height = 960,720
    aspect_ratio = cam_width / cam_height
    fov = 60
    nearVal = 0.01
    farVal = 100

    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, nearVal, farVal)

    img = p.getCameraImage(cam_width, cam_height, view_matrix, proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL)

    # rgb_array = np.array(rgbPixels, dtype = np.uint8) # 8 bit integers
    # rgb_array = np.reshape(rgb_array, (cam_height, cam_width, 4))
    # depth_array = np.reshape(depthPixels, [cam_height, cam_width])

    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 
    p.stepSimulation()
    time.sleep(1/240)

p.disconnect()

