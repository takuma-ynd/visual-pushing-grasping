import argparse
import numpy as np
from phys_intuition_env import PhysIntuitionEnv
from phys_wrapper import PhysPushActionWrapper

'''
Y
^
0          rotation
|
|              8
|              ^
|   9, 10... 15|1, 2...7
|              |
|              0
223
O----------------------223>  X
        [Robot]

'''
def main(args):
    print('starting env..')
    physenv = PhysPushActionWrapper(PhysIntuitionEnv(args))
    # import ipdb; ipdb.set_trace()
    # for arg in args.keys():
    #     print(arg, '=', getattr(args, arg))
    obs = physenv.reset()
    done = False
    action_list = []
    for binary in [-1, 1]:
        for binary2 in [-1, 1]:
            for binary3 in [-1, 1]:
                for binary4 in [-1, 1]:
                    action_list.append( [binary, binary2, binary3, binary4] )
    while not done:
        # action = physenv.action_space.sample()
        action = action_list.pop()
        # print('action', action)
        # best_pix = np.asarray([float(input('type input:')) for _ in range(3)], dtype=int).reshape(-1)
        # action = np.asarray([float(input('type input:')) for _ in range(4)], dtype=np.float32).reshape(-1)
        print('action', action)
        # print('action', action)
        obs, reward, done, _ =  physenv.step(action)
        print('reward', reward)
        # img = list(obs.values())[0]


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,                           help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,                           help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='reinforcement',                               help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=False,                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,              help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,          help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')
    
    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')
    
    parser.add_argument('--remote_api_port', dest='remote_api_port', type=int, action='store', default=19997,          help='remote api port')
    parser.add_argument('--vrep-dir', type=str, required=True,
                        help='Directory path to vrep')
    parser.add_argument('--sim-path', type=str, required=True,
                        help='File path to simulation.ttt file')
    parser.add_argument('--display', type=str, default=None)
    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
