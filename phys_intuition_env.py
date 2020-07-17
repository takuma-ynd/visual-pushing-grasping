# template source: https://github.com/HendrikPN/gym-template/blob/master/gym_foo/envs/foo_env.py
import os
import subprocess
import time
import logging
import cv2
from robot import Robot
from logger import Logger
import numpy as np
import random
import gym
from gym import error, spaces, utils as gym_utils
from gym.utils import seeding
import heuristics
import utils
from mypytools import timed
actstr2id = {'push':0, 'grasp':1}
id2actstr = ['push', 'grasp']
class SharedVars():
    no_change_count = None
    prev_primitive_action = None
    # ', 'prev_color_img', 'prev_depth_img', 'prev_color_heightmap', 'prev_depth_heightmap', 'prev_best_pix_ind', 'prev_obj2segmentation_img', 'cam_depth_scale'])
    
class SharedObs():
    # SharedObs = namedtuple('SharedObs', ['valid_depth_heightmap'])
    valid_depth_heightmap = None

def calc_reward(robot, no_change_count, prev_obj_positions):
    reward = 0
    if no_change_count[0] != 0 or no_change_count[1] != 0:
        reward = 0
    else:
        obj_positions = np.asarray(robot.get_obj_positions())
        prev_dist = np.sqrt(np.sum(np.power(prev_obj_positions[0] - robot.workspace_limits[:, 0], 2), axis=0))
        dist = np.sqrt(np.sum(np.power(obj_positions[0] - robot.workspace_limits[:, 0], 2), axis=0))
        # max_dist = np.sqrt(np.sum(np.power(self.workspace_limits[:, 1] - self.workspace_limits[:, 0], 2), axis=0))
        # print('prev_dist', prev_dist)
        # print('dist', dist)
        reward = (prev_dist - dist) * 10

    return reward
    

class PhysIntuitionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, args, lock=None):
        """
        Every environment should be derived from gym.Env and at least contain the variables observation_space and action_space 
        specifying the type of possible observations and actions using spaces.Box or spaces.Discrete.

        Example:
        >>> EnvTest = FooEnv()
        >>> EnvTest.observation_space=spaces.Box(low=-1, high=1, shape=(3,4))
        >>> EnvTest.action_space=spaces.Discrete(2)
        """
        self.local_counter = 0
        self.reset_counter = 0
        self.iteration = 0
        is_sim = args.is_sim # Run in simulation?
        self.is_sim = is_sim
        self.reset_threshold = getattr(args, 'reset_threshold', 10)
        obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
        num_obj = args.num_obj if is_sim else None # Number of objects to add to simulation
        tcp_host_ip = args.tcp_host_ip if not is_sim else None # IP and port to robot arm as TCP client (UR5)
        tcp_port = args.tcp_port if not is_sim else None
        rtc_host_ip = args.rtc_host_ip if not is_sim else None # IP and port to robot arm as real-time client (UR5)
        rtc_port = args.rtc_port if not is_sim else None
        if is_sim:
            self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
            # self.workspace_limits = np.asarray([[-0.524, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        else:
            self.workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        self.heightmap_resolution = args.heightmap_resolution # Meters per pixel of heightmap
        random_seed = args.random_seed
        remote_api_port = args.remote_api_port
        self.remote_api_port = remote_api_port
        self.logger = logging.getLogger('PhysIntuitionEnv')
        # force_cpu = args.force_cpu

        # ------------- Algorithm options -------------
        # method = args.method # 'reactive' (supervised learning) or 'reinforcement' (reinforcement learning ie Q-learning)
        # push_rewards = args.push_rewards if method == 'reinforcement' else None  # Use immediate rewards (from change detection) for pushing?
        # future_reward_discount = args.future_reward_discount
        # experience_replay = args.experience_replay # Use prioritized experience replay?
        # heuristic_bootstrap = args.heuristic_bootstrap # Use handcrafted grasping algorithm when grasping fails too many times in a row?
        # explore_rate_decay = args.explore_rate_decay
        # grasp_only = args.grasp_only

        # -------------- Testing options --------------
        is_testing = args.is_testing
        self.is_testing = is_testing
        max_test_trials = args.max_test_trials # Maximum number of test runs per case/scenario
        test_preset_cases = args.test_preset_cases
        test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None


        # ------ Pre-loading and logging options ------
        # load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
        # snapshot_file = os.path.abspath(args.snapshot_file)  if load_snapshot else None
        continue_logging = args.continue_logging # Continue logging from previous session
        logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
        # save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True

        self.sleeptime_before_bootup = getattr(args, 'sleeptime_before_bootup', 2)

        # ------ Others ------
        self.num_rotations = 16


        # Set random seed
        self.seed(random_seed)

        # run vrep with subprocess call
        vrep_path = os.path.join(args.vrep_dir, 'vrep.sh')
        sim_path = args.sim_path
        # pythonpath = 'PYTHONPATH={}'.format(args.vrep_dir)
        xvfb = ['xvfb-run', '--auto-servernum', '-s', '-screen 0, 640x480x24 -extension RANDR']

        envvars = os.environ.copy()
        # envvars['PYTHONPATH'] = args.vrep_dir + ":" + envvars['PYTHONPATH']
        envvars['LD_LIBRARY_PATH'] = args.vrep_dir + ":" + envvars.get('LD_LIBRARY_PATH', '')
        # Synchronization
        if lock is not None:
            lock.acquire()
            self.logger.info('process is acquired (port: {})'.format(remote_api_port))

        try:
            self.logger.debug('modifying remote api port...')
            with utils.modified_remote_api_port(args.vrep_dir, remote_api_port):
                with open(os.path.join(args.vrep_dir, 'remoteApiConnections.txt'), 'r') as f:
                    print(f.read())
                if args.display is not None:
                    envvars['DISPLAY'] = args.display
                    command = [vrep_path, sim_path]
                else:
                    command = [*xvfb, vrep_path, '-h', sim_path]
                self.logger.debug('command: {}'.format(command))
                self.logger.debug('Launching simulator...')

                # show logs from the simulator if logging level is DEBUG
                stdout = None if self.logger.level == logging.DEBUG else subprocess.DEVNULL
                subprocess.Popen(command, env=envvars, stdout=stdout)
                # subprocess.Popen(command, env=envvars)
                self.logger.debug('restoring remote api port...')
                time.sleep(self.sleeptime_before_bootup)  # sleep for a bit until vrep boots up
        except Exception as e:
            # restore remoteApiConnections.txt
            print('restoring remoteApiConnections.txt...')
            subprocess.Popen(['cp', os.path.join(args.vrep_dir, 'remoteApiConnections.txt.backup'), os.path.join(args.vrep_dir, 'remoteApiConnections.txt')])
            raise e

        # Initialize pick-and-place system (camera and robot)
        print('instantiating Robot class with port {}...'.format(remote_api_port))
        robot = Robot(is_sim, obj_mesh_dir, num_obj, self.workspace_limits,
                      tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                      is_testing, test_preset_cases, test_preset_file, remote_api_port)
        self.robot = robot

        if lock is not None:
            lock.release()
            self.logger.info('released the process (port: {})'.format(remote_api_port))

        # Initialize data logger
        data_logger = Logger(continue_logging, logging_directory)
        data_logger.save_camera_info(self.robot.cam_intrinsics, self.robot.cam_pose, self.robot.cam_depth_scale) # Save camera intrinsics and pose
        data_logger.save_heightmap_info(self.workspace_limits, args.heightmap_resolution) # Save heightmap parameters# 
        self.data_logger = data_logger

        # SharedVars = namedtuple('SharedVars', ['no_change_count', 'prev_primitive_action', 'prev_color_img', 'prev_depth_img', 'prev_color_heightmap', 'prev_depth_heightmap', 'prev_best_pix_ind', 'prev_obj2segmentation_img', 'cam_depth_scale'])
        self.shared_vars = SharedVars()

        # SharedObs = namedtuple('SharedObs', ['valid_depth_heightmap'])
        self.shared_obs = SharedObs()

        # lows_yx = self.workspace_limits[:, 0][:2][::-1]
        # lows_angle = [0.0]
        # low = np.asarray(lows_angle + lows_yx)
        # highs_yx = self.workspace_limits[:, 1][:2][::-1]
        # highs_angle = [16.0]
        # high = np.asarray(highs_angle + highs_yx)
        # self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Box(low=np.array([0, 0, 0]), high=np.array([16, 224 - 1, 224 - 1]), dtype=int)))
        self.n_actions = 4  # dir_x, dir_y, x, y
        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')
        # TEMP:
        # self.action_space = spaces.Box(low=np.array([0, 0, 0], dtype=int), high=np.array([16, 224 - 1, 224 - 1], dtype=int), dtype=int)
        self.observation_space = spaces.Box(low=np.zeros((480, 640, 3), dtype=np.uint8), high=np.ones((480, 640, 3), dtype=np.uint8))

    def seed(self, seed_val):
        random.seed(seed_val)
        np.random.seed(seed_val)


    def _get_action(self, raw_action, primitive_action, valid_depth_heightmap):
        assert len(raw_action) == 4, 'len(raw_action): {}'.format(len(raw_action))  # dir_x, dir_y, x, y
        raw_direction, raw_position = raw_action[:2], raw_action[2:]
        x_width = self.workspace_limits[0][1] - self.workspace_limits[0][0]
        y_width = self.workspace_limits[1][1] - self.workspace_limits[1][0]
        x_min = self.workspace_limits[0][0]
        y_min = self.workspace_limits[1][0]
        z_min = self.workspace_limits[2][0]
        action_width = self.action_space.high[-1] - self.action_space.low[-1]
        min_action = self.action_space.low[0]
        normalized_action = (raw_position - min_action) / action_width
        best_pix_x = int(normalized_action[0] * (valid_depth_heightmap.shape[0] - 1))
        best_pix_y = int(normalized_action[1] * (valid_depth_heightmap.shape[1] - 1))
        primitive_position = [normalized_action[0] * x_width + x_min, normalized_action[1] * y_width + y_min, valid_depth_heightmap[best_pix_y][best_pix_x] + z_min]

        # Handle the corner cases
        if raw_direction[0] == 0:
            if random.random() < 0.5:
                raw_direction[0] = 1e-8
            else:
                raw_direction[0] = - 1e-8

        if raw_direction[1] == 0:
            if random.random() < 0.5:
                raw_direction[1] = 1e-8
            else:
                raw_direction[1] = - 1e-8

        if raw_direction[0] > 0 and raw_direction[1] > 0:  # First quadrant
            best_rotation_angle = np.arctan(raw_direction[1] / raw_direction[0])
        elif raw_direction[0] < 0 and raw_direction[1] < 0:  # Third quadrant
            best_rotation_angle = np.arctan(raw_direction[1] / raw_direction[0]) + np.pi
        elif raw_direction[0] > 0 and raw_direction[1] < 0:  # Fourth quadrant
            best_rotation_angle = np.arctan(raw_direction[1] / raw_direction[0]) + 2 * np.pi
        elif raw_direction[0] < 0 and raw_direction[1] > 0:  # Second quadrant
            best_rotation_angle = np.arctan(raw_direction[1] / raw_direction[0]) + np.pi

        # If pushing, adjust start position, and make sure z value is safe and not too low
        if id2actstr[primitive_action] == 'push': # or nonlocal_variables['primitive_action'] == 'place':
            finger_width = 0.02
            safe_kernel_width = int(np.round((finger_width/2)/self.heightmap_resolution))
            local_region = valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]), max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1])]
            if local_region.size == 0:
                safe_z_position = self.workspace_limits[2][0]
            else:
                safe_z_position = np.max(local_region) + self.workspace_limits[2][0]
            primitive_position[2] = safe_z_position

        return primitive_position, best_rotation_angle

    def _get_action2(self, raw_action, primitive_action, valid_depth_heightmap):
        assert raw_action.shape == (3,)  # radian, x, y
        raw_rotation, raw_position = raw_action[0], raw_action[1:]
        x_width = self.workspace_limits[0][1] - self.workspace_limits[0][0]
        y_width = self.workspace_limits[1][1] - self.workspace_limits[1][0]
        x_min = self.workspace_limits[0][0]
        y_min = self.workspace_limits[1][0]
        z_min = self.workspace_limits[2][0]
        action_width = self.action_space.high[-1] - self.action_space.low[-1]
        min_action = self.action_space.low[0]
        normalized_action = (raw_position - min_action) / action_width
        best_pix_x = int(normalized_action[0] * valid_depth_heightmap.shape[0])
        best_pix_y = int(normalized_action[1] * valid_depth_heightmap.shape[1])
        primitive_position = [normalized_action[0] * x_width + x_min, normalized_action[1] * y_width + y_min, valid_depth_heightmap[best_pix_y][best_pix_x] + z_min]

        best_rotation_angle = raw_rotation % np.pi

        # If pushing, adjust start position, and make sure z value is safe and not too low
        if id2actstr[primitive_action] == 'push': # or nonlocal_variables['primitive_action'] == 'place':
            finger_width = 0.02
            safe_kernel_width = int(np.round((finger_width/2)/self.heightmap_resolution))
            local_region = valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]), max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1])]
            if local_region.size == 0:
                safe_z_position = self.workspace_limits[2][0]
            else:
                safe_z_position = np.max(local_region) + self.workspace_limits[2][0]
            primitive_position[2] = safe_z_position

        return primitive_position, best_rotation_angle

    def get_action(self, best_pix_ind, primitive_action, num_rotations, valid_depth_heightmap):
        '''calculate primitive_position, best_rotation_angle from best_pix_ind'''

        # Compute 3D position of pixel
        # print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], best_pix_ind[0], best_pix_ind[1], best_pix_ind[2]))
        best_rotation_angle = np.deg2rad(best_pix_ind[0]*(360.0/num_rotations))
        best_pix_x = best_pix_ind[2]
        best_pix_y = best_pix_ind[1]
        primitive_position = [best_pix_x * self.heightmap_resolution + self.workspace_limits[0][0], best_pix_y * self.heightmap_resolution + self.workspace_limits[1][0], valid_depth_heightmap[best_pix_y][best_pix_x] + self.workspace_limits[2][0]]

        # If pushing, adjust start position, and make sure z value is safe and not too low
        if id2actstr[primitive_action] == 'push': # or nonlocal_variables['primitive_action'] == 'place':
            finger_width = 0.02
            safe_kernel_width = int(np.round((finger_width/2)/self.heightmap_resolution))
            local_region = valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]), max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1])]
            if local_region.size == 0:
                safe_z_position = self.workspace_limits[2][0]
            else:
                safe_z_position = np.max(local_region) + self.workspace_limits[2][0]
            primitive_position[2] = safe_z_position

        return primitive_position, best_rotation_angle

    def get_heuristic_action(self, primitive_action, valid_depth_heightmap):
        # If heuristic bootstrapping is enabled: if change has not been detected more than 2 times, execute heuristic algorithm to detect grasps/pushes
        # NOTE: typically not necessary and can reduce final performance.
        if id2actstr[primitive_action] == 'push':
            # print(
            #     'Change not detected for more than two pushes. Running heuristic pushing.')
            print('Running heuristic pushing...')
            best_pix_ind = heuristics.push_heuristic(valid_depth_heightmap)
            self.shared_vars.no_change_count[0] = 0
        elif id2actstr[primitive_action] == 'grasp':
            # print(
            #     'Change not detected for more than two grasps. Running heuristic grasping.')
            print('Running heuristic grasping.')
            best_pix_ind = heuristics.grasp_heuristic(valid_depth_heightmap)
            self.shared_vars.no_change_count[1] = 0

        # return self.get_action(best_pix_ind, primitive_action, self.num_rotations, valid_depth_heightmap)
        action = (primitive_action, best_pix_ind)
        return action



    def step(self, action):
        """
        This method is the primary interface between environment and agent.

        Paramters: 
            action: int
                    the index of the respective action (if action space is discrete)

        Returns:
            output: (array, float, bool)
                    information provided by the environment about its current state:
                    (observation, reward, done)
        """
        with timed('step process'):
            done = False
            # primitive_action, best_pix_ind = action
            primitive_action, raw_action = action
            # print('action', action)
            # print('primitive_action', primitive_action)
            # print('best_pix_ind', best_pix_ind)
            # primitive_position, best_rotation_angle = self.get_action(best_pix_ind, primitive_action, self.num_rotations, self.shared_obs.valid_depth_heightmap)
            primitive_position, best_rotation_angle = self._get_action(raw_action, primitive_action, self.shared_obs.valid_depth_heightmap)
            # primitive_action, primitive_position, best_rotation_angle = action

            # Execute primitive
            if id2actstr[primitive_action] == 'push':
                push_success = self.robot.push(primitive_position, best_rotation_angle, self.workspace_limits)
                # print('Push successful: %r' % (push_success))
            elif id2actstr[primitive_action] == 'grasp':
                grasp_success = self.robot.grasp(primitive_position, best_rotation_angle, self.workspace_limits)
                # print('Grasp successful: %r' % (grasp_success))


            # Get latest RGB-D image
            color_img, depth_img = self.robot.get_camera_data()
            depth_img = depth_img * self.robot.cam_depth_scale # Apply depth scale from calibration
            segment_img = self.robot.get_segmentation_camera_data()
            obj2segmented_img = self.robot.get_segmented_images(segment_img, color_img)
            color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, self.robot.cam_intrinsics, self.robot.cam_pose, self.workspace_limits, self.heightmap_resolution)
            valid_depth_heightmap = depth_heightmap.copy()
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

            # self.data_logger.save_images(self.iteration, color_img, depth_img, '0', reset_counter=self.reset_counter)
            # self.data_logger.save_heightmaps(self.iteration, color_heightmap, valid_depth_heightmap, '0', reset_counter=self.reset_counter)
            # self.data_logger.save_segmented_images(self.iteration, self.local_counter-1, obj2segmented_img, reset_counter=self.reset_counter)

            # TEMP:
            obs = list(obj2segmented_img.values())[0]


            # Detect changes
            depth_diff = abs(depth_heightmap - self.shared_vars.prev_depth_heightmap)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.01] = 0
            depth_diff[depth_diff > 0] = 1
            change_threshold = 300
            change_value = np.sum(depth_diff)
            change_detected = change_value > change_threshold  # or prev_grasp_success
            # print('Change detected: %r (value: %d)' % (change_detected, change_value))
            # print('no_change count:', self.shared_vars.no_change_count)

            if change_detected:
                if id2actstr[self.shared_vars.prev_primitive_action] == 'push':
                    self.shared_vars.no_change_count[0] = 0
                elif id2actstr[self.shared_vars.prev_primitive_action] == 'grasp':
                    self.shared_vars.no_change_count[1] = 0
            else:
                if id2actstr[self.shared_vars.prev_primitive_action] == 'push':
                    self.shared_vars.no_change_count[0] += 1
                elif id2actstr[self.shared_vars.prev_primitive_action] == 'grasp':
                    self.shared_vars.no_change_count[1] += 1
            # print('no_change_count:', self.shared_vars.no_change_count)


            # ===== reset condition ====
            # Reset simulation or pause real-world training if table is empty
            stuff_count = np.zeros(valid_depth_heightmap.shape)
            stuff_count[valid_depth_heightmap > 0.02] = 1
            empty_threshold = 300
            if self.is_sim and self.is_testing:
                empty_threshold = 10
            if np.sum(stuff_count) < empty_threshold or (self.is_sim and self.shared_vars.no_change_count[0] + self.shared_vars.no_change_count[1] > 10): # or counter % n_trials == 0:
                print('Not enough objects in view (value: %d / threshold: %d)! Repositioning objects.' % (np.sum(stuff_count), empty_threshold))
                done = True
            if (self.shared_vars.no_change_count[0] > self.reset_threshold) or (self.shared_vars.no_change_count[1] >= self.reset_threshold):
                done = True



            # reward = 0
            # reward = self.robot.get_my_task_score()
            reward = calc_reward(self.robot, self.shared_vars.no_change_count, self.shared_vars.prev_obj_positions)

            # Save information for next training step
            self.shared_vars.prev_color_img = color_img.copy()
            self.shared_vars.prev_depth_img = depth_img.copy()
            self.shared_vars.prev_obj2segmentation_img = list({key: img.copy() for key, img in obj2segmented_img.items()}.values())[0]
            self.shared_vars.prev_color_heightmap = color_heightmap.copy()
            self.shared_vars.prev_depth_heightmap = depth_heightmap.copy()
            # prev_valid_depth_heightmap = valid_depth_heightmap.copy()
            # prev_push_success = nonlocal_variables['push_success']
            # prev_grasp_success = nonlocal_variables['grasp_success']
            self.shared_vars.prev_primitive_action = primitive_action
            # prev_push_predictions = push_predictions.copy()
            # prev_grasp_predictions = grasp_predictions.copy()
            # self.shared_vars.prev_best_pix_ind = best_pix_ind
            self.shared_obs.valid_depth_heightmap = valid_depth_heightmap
            self.shared_vars.prev_obj_positions = self.robot.get_obj_positions().copy()

            self.logger.info('reward: {}'.format(reward))
            # self.logger.info('--- iteration {} is done ---'.format(self.iteration))
            self.iteration += 1
            self.local_counter += 1

        return obs, reward, done, {}

    def reset(self):
        """
        This method resets the environment to its initial values.

        Returns:
            observation:    array
                            the initial state of the environment
        """
        if self.is_sim:
            if self.reset_counter != 0:  # Don't call them at the first time to call reset
                self.logger.debug('restarting sim...')
                self.robot.restart_sim()
                self.logger.debug('adding objects...')
                self.robot.add_objects()
            # counter = 0
            # reset_counter += 1
            # if is_testing: # If at end of test run, re-load original weights (before test run)
            #     print('loading model...')
            #     trainer.model.load_state_dict(torch.load(snapshot_file))
            #     print('finished...')

            # Get latest RGB-D image
            color_img, depth_img = self.robot.get_camera_data()
            depth_img = depth_img * self.robot.cam_depth_scale # Apply depth scale from calibration
            segment_img = self.robot.get_segmentation_camera_data()
            obj2segmented_img = self.robot.get_segmented_images(segment_img, color_img)
            color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, self.robot.cam_intrinsics, self.robot.cam_pose, self.workspace_limits, self.heightmap_resolution)
            valid_depth_heightmap = depth_heightmap.copy()
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

            # self.data_logger.save_images(self.iteration, color_img, depth_img, '0', reset_counter=self.reset_counter)
            # self.data_logger.save_heightmaps(self.iteration, color_heightmap, valid_depth_heightmap, '0', reset_counter=self.reset_counter)
            # self.data_logger.save_segmented_images(self.iteration, self.local_counter-1, obj2segmented_img, reset_counter=self.reset_counter)
            obs = list(obj2segmented_img.values())[0]


            # Save information for next training step
            self.shared_vars.prev_color_img = color_img.copy()
            self.shared_vars.prev_depth_img = depth_img.copy()
            self.shared_vars.prev_obj2segmentation_img = {key: img.copy() for key, img in obj2segmented_img.items()}
            self.shared_vars.prev_color_heightmap = color_heightmap.copy()
            self.shared_vars.prev_depth_heightmap = depth_heightmap.copy()
            # prev_valid_depth_heightmap = valid_depth_heightmap.copy()
            # prev_push_success = nonlocal_variables['push_success']
            # prev_grasp_success = nonlocal_variables['grasp_success']
            self.shared_vars.prev_primitive_action = actstr2id['push']
            # prev_push_predictions = push_predictions.copy()
            # prev_grasp_predictions = grasp_predictions.copy()
            # self.shared_vars.prev_best_pix_ind = best_pix_ind
            self.shared_vars.prev_obj_positions = self.robot.get_obj_positions().copy()

            self.shared_obs.valid_depth_heightmap = valid_depth_heightmap


        else:
            print('Not enough stuff on the table ! Pausing for 30 seconds.')
            time.sleep(30)
            print('Not enough stuff on the table ! Flipping over bin of objects...')
            self.robot.restart_real()

            # Get latest RGB-D image
            color_img, depth_img = self.robot.get_camera_data()
            depth_img = depth_img * self.robot.cam_depth_scale # Apply depth scale from calibration
            segment_img = self.robot.get_segmentation_camera_data()
            obj2segmented_img = self.robot.get_segmented_images(segment_img, color_img)
            color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, self.robot.cam_intrinsics, self.robot.cam_pose, self.workspace_limits, self.heightmap_resolution)
            valid_depth_heightmap = depth_heightmap.copy()
            valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

            # self.data_logger.save_images(self.iteration, color_img, depth_img, '0', reset_counter=self.reset_counter)
            # self.data_logger.save_heightmaps(self.iteration, color_heightmap, valid_depth_heightmap, '0', reset_counter=self.reset_counter)
            # self.data_logger.save_segmented_images(self.iteration, self.local_counter-1, obj2segmented_img, reset_counter=self.reset_counter)
            obs = list(obj2segmented_img.values())[0]


            # Save information for next training step
            self.shared_vars.prev_color_img = color_img.copy()
            self.shared_vars.prev_depth_img = depth_img.copy()
            self.shared_vars.prev_obj2segmentation_img = {key: img.copy() for key, img in obj2segmented_img.items()}
            self.shared_vars.prev_color_heightmap = color_heightmap.copy()
            self.shared_vars.prev_depth_heightmap = depth_heightmap.copy()
            # prev_valid_depth_heightmap = valid_depth_heightmap.copy()
            # prev_push_success = nonlocal_variables['push_success']
            # prev_grasp_success = nonlocal_variables['grasp_success']
            # self.shared_vars.prev_primitive_action = primitive_action
            # prev_push_predictions = push_predictions.copy()
            # prev_grasp_predictions = grasp_predictions.copy()
            # self.shared_vars.prev_best_pix_ind = best_pix_ind
            self.shared_vars.prev_obj_positions = self.robot.get_obj_positions().copy()

            self.shared_obs.valid_depth_heightmap = valid_depth_heightmap

        # trainer.clearance_log.append([trainer.iteration]) 
        # self.data_logger.write_to_log('clearance', trainer.clearance_log)

        self.reset_counter += 1
        self.local_counter = 0
        self.shared_vars.no_change_count = [0, 0]
        return obs

    def render(self, mode='human', close=False):
        """
        This methods provides the option to render the environment's behavior to a window 
        which should be readable to the human eye if mode is set to 'human'.
        """
        pass

