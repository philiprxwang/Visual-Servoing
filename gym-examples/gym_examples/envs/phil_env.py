import pybullet as p
import pybullet_data
import math
import time
import os
import numpy as np
import random
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

IMG_HEIGHT = 720
IMG_WIDTH = 960
NUM_CHANNELS = 3
NUM_FRAMES = 1
DIR_SIZE = 0.1

class PhilEnv(gym.Env):
    """
    Version History
    v1: Observation is RGB array from camera 
    v0: Initial version - observation was end effector and target pos
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        
        self.height = IMG_HEIGHT
        self.width = IMG_WIDTH
        self.num_channels = NUM_CHANNELS
        self.num_frames = NUM_FRAMES
        self.dir_size = DIR_SIZE

        p.connect(p.GUI)
        #p.connect(p.DIRECT) 
        self.action_space = spaces.Discrete(6) # total 6 actions: front, back, left, right, up, down
        self.observation_space = spaces.Box(low=0, high = 255, shape = (self.num_channels, self.height, self.width), dtype = np.uint8) # observation is RGB array
        
        """
        action_to_direction dict maps action to movement direction of robot arm.
        0,1,2,3,4,5 corresponds to front, back, left, right, up, down respectively
        """
        
        self._action_to_direction = {
            0: np.array([self.dir_size, 0, 0]),
            1: np.array([-self.dir_size, 0, 0]),
            2: np.array([0, self.dir_size, 0]),
            3: np.array([0, -self.dir_size, 0]),
            4: np.array([0, 0, self.dir_size]),
            5: np.array([0, 0, -self.dir_size])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed = None, options = None):
        p.resetSimulation()
        p.setGravity(0,0,-9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        planeId = p.loadURDF('plane.urdf')
        self.pandaId = p.loadURDF('franka_panda/panda.urdf', basePosition=[0,0,0.6],baseOrientation= p.getQuaternionFromEuler([0,0,0]),useFixedBase=True)
        tableId = p.loadURDF('table/table.urdf', basePosition=[0.65/2,0,0], baseOrientation=p.getQuaternionFromEuler([0,0, 0]))
        self.cubeId = p.loadURDF('cube_small.urdf', basePosition = [0.64, 0, 0.79], globalScaling = 0.6)
        p.changeDynamics(self.cubeId, -1, 5) # changing the mass of the cube makes it more stationary when applying fixed constraint later on
        neutral_joint_values = [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00]
        for i in range(len(neutral_joint_values)):  
            p.resetJointState(self.pandaId, i, targetValue = neutral_joint_values[i]) # reset panda to neutral pos 

        self.rand_objId = p.loadURDF('random_urdfs/000/000.urdf', basePosition = [0.55,0, 0.61]) # some random object
        
        cube_orn = p.getQuaternionFromEuler([0,0,0])
        panda_cid = p.createConstraint(self.pandaId, 11, self.cubeId, -1, p.JOINT_FIXED, [0,0,0], [0.035, 0, -0.03], childFramePosition = [0,0,0], childFrameOrientation = cube_orn)

        observation = self._get_obs() # returns rgb array
        info = self._get_info() # returns cartesian dis

        if self.render_mode == 'human':
            self.render()
            
        return observation, info
    
    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        self.current_pos = np.array(p.getLinkState(self.pandaId, 11)[0]) # state of the end effector
        direction = self._action_to_direction[action]
        self.current_pos = self.current_pos + direction
        self.current_pos[0] = np.clip(self.current_pos[0], 0, 5) # clip x
        self.current_pos[1] = np.clip(self.current_pos[1], -3, 3) # clip y
        self.current_pos[2] = np.clip(self.current_pos[2], 0.6, 5) # clip z: gripper to not touch table

        target_orn = p.getQuaternionFromEuler([0,-math.pi, math.pi/2]) # downward orientation
        joint_poses = p.calculateInverseKinematics(self.pandaId, 11, self.current_pos, target_orn)
        for i in range(7): # 7 DoF
            p.setJointMotorControl2(self.pandaId, i, controlMode=p.POSITION_CONTROL, targetPosition=joint_poses[i])

        observation = self._get_obs() # returns rgb array
        info = self._get_info() 
        
        if info['distance'] <= 0.05:
            # terminate because promixity reached
            reward = 100
            terminated = True 
        elif info['distance'] <= 0.5:
            # reward that is inversely proportional to distance
            k = 0.4
            reward = 1 / (k * info['distance']) 
            terminated = False
        elif info['distance'] == 3:
            reward = -1000 # terminate because gripper was too far
            terminated = True  
        else:
            reward = -1 # small negative reward for every step 
            terminated = False

        if self.render_mode == 'human':
            self.render()

        p.stepSimulation()
        # time.sleep(1/240)

        return observation, reward, terminated, False, info

    def _get_info(self):
        rand_obj_pos = p.getBasePositionAndOrientation(self.rand_objId)[0]
        end_effector_pos = p.getLinkState(self.pandaId, 11)[0]
        cart_dis = np.sqrt((rand_obj_pos[0] - end_effector_pos[0])**2+(rand_obj_pos[1] - end_effector_pos[1])**2+(rand_obj_pos[2] - end_effector_pos[2])**2)
        return {'distance': cart_dis}
    
    def _get_obs(self):
        observation = np.zeros((self.num_channels, self.height, self.width), dtype=np.uint8)
    #    for i in range(self.num_frames):
    #         observation[:, :, :, i] = np.array(self.render())
    #         return {'camera_rgb': observation}
        channel_first_array = self.render().transpose(2, 0, 1)
        observation = np.array(channel_first_array, dtype=np.uint8)
        return observation

    def render(self, mode = 'human'):
        cube_orn = p.getQuaternionFromEuler([0,0,0])

        # panda_cid used to be here 

        cube_state = p.getBasePositionAndOrientation(self.cubeId)
        cube_pos = np.array(cube_state[0])
        cube_orn = np.array(cube_state[1])
        rot_matrix = np.array(p.getMatrixFromQuaternion(cube_orn)).reshape(3,3) # 3x3 rotation matrix (right, forward, up by columns)
        forward_vec = rot_matrix.dot(np.array((0, 0, 1)))
        up_vec = rot_matrix.dot(np.array((0, 1, 0)))

        target_position = cube_pos + 0.1 * forward_vec

        view_matrix = p.computeViewMatrix(cube_pos, target_position, up_vec)

        aspect_ratio = self.width / self.height
        fov = 60
        nearVal = 0.01
        farVal = 100

        proj_matrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, nearVal, farVal)

        self.rgb_img = p.getCameraImage(self.width, self.height, view_matrix, proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL)[2]
        self.rgb_img = np.array(self.rgb_img).reshape(self.height, self.width, 4)[:, :, :3]

        return self.rgb_img

    def close(self):
        p.disconnect()


