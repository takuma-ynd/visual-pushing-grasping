"""A training script of Soft Actor-Critic on RoboschoolAtlasForwardWalk-v1."""
import argparse
import functools
import logging
import os
import sys
import time
from multiprocessing import Lock

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import gym
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl import experiments
from chainerrl import misc
from chainerrl import replay_buffer

from config import Config
from phys_intuition_env import PhysIntuitionEnv
from phys_wrapper import PhysPushActionWrapper
from multiprocess_vector_env import MultiprocessVectorEnv
import my_vgg

class TransposeObservation(gym.ObservationWrapper):
    """Transpose observations."""

    def __init__(self, env, axes):
        super().__init__(env)
        self._axes = axes
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.transpose(*self._axes),
            high=env.observation_space.high.transpose(*self._axes),
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(*self._axes)

def concat_obs_and_action(obs, action):
    """Concat observation and action to feed the critic."""
    return F.concat((obs.reshape((action.shape[0], -1)), action), axis=-1)

def make_env(args, seed, idx, test, lock=None):
    # from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv  # NOQA
    # Use different random seeds for train and test envs
    process_seed = int(seed)
    env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
    # Set a random seed for this subprocess
    misc.set_random_seed(env_seed)
    # env = KukaDiverseObjectEnv(
    #     isDiscrete=True,
    #     renders=args.render and (args.demo or not test),
    #     height=84,
    #     width=84,
    #     maxSteps=max_episode_steps,
    #     isTest=test,
    # )

    config_vars = Config()
    print('-------------------------------------- MAKE ENV ------------------------------------')
    if test:
        config_vars.remote_api_port = 19997 - 256 + idx
    else:
        config_vars.remote_api_port = 19997 + idx
    config_vars.vrep_dir = args.vrep_dir
    config_vars.sim_path = args.sim_path
    print('port:', config_vars.remote_api_port)
    config_vars.sleeptime_before_bootup = args.sleeptime_before_bootup
    config_vars.display = args.display
    env = PhysIntuitionEnv(config_vars, lock)
    env = gym.wrappers.ResizeObservation(env, (224, 224))
    # assert env.observation_space is None
    # env.observation_space = gym.spaces.Box(
    #     low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
    # (84, 84, 3) -> (3, 84, 84)
    env = TransposeObservation(env, (2, 0, 1))
    # env = ObserveElapsedSteps(env, max_episode_steps)
    # KukaDiverseObjectEnv internally asserts int actions
    # env = CastAction(env, int)
    env.seed(int(env_seed))

    env = chainerrl.wrappers.CastObservationToFloat32(env)  # TEMP
    # Normalize action space to [-1, 1]^n
    # env = chainerrl.wrappers.NormalizeActionSpace(env)
    env = PhysPushActionWrapper(env)

    # if test and args.record:
    #     assert args.render,\
    #         'To use --record, --render needs be specified.'
    #     video_dir = os.path.join(args.outdir, 'video_{}'.format(idx))
    #     os.mkdir(video_dir)
    #     env = RecordMovie(env, video_dir)
    return env

# def make_env(args, seed, idx, test):
#     # env = gym.make(args.env)
#     config_vars = Config()
#     if test:
#         config_vars.remote_api_port = 19998
#     else:
#         config_vars.remote_api_port = 19997
#     config_vars.vrep_dir = args.vrep_dir
#     config_vars.sim_path = args.sim_path
#     env = PhysActionWrapper(PhysIntuitionEnv(config_vars))
#     # Use different random seeds for train and test envs
#     env_seed = 2 ** 32 - 1 - seed if test else seed
#     env.seed(int(env_seed))
#     # Cast observations to float32 because our model uses float32
#     env = chainerrl.wrappers.CastObservationToFloat32(env)  # TEMP
#     # Normalize action space to [-1, 1]^n
#     env = chainerrl.wrappers.NormalizeActionSpace(env)

#     if args.monitor:
#         env = chainerrl.wrappers.Monitor(env, args.outdir)
#     # if isinstance(env.action_space, spaces.Box):
#     #     misc.env_modifiers.make_action_filtered(env, clip_action_filter)
#     if ((args.render_eval and test) or
#             (args.render_train and not test)):
#         env = chainerrl.wrappers.Render(env)
#     return env

# def make_env(args, seed, test):
#     if args.env.startswith('Roboschool'):
#         # Check gym version because roboschool does not work with gym>=0.15.6
#         from distutils.version import StrictVersion
#         gym_version = StrictVersion(gym.__version__)
#         if gym_version >= StrictVersion('0.15.6'):
#             raise RuntimeError('roboschool does not work with gym>=0.15.6')
#         import roboschool  # NOQA
#     env = gym.make(args.env)
#     # Unwrap TimiLimit wrapper
#     assert isinstance(env, gym.wrappers.TimeLimit)
#     env = env.env
#     # Use different random seeds for train and test envs
#     env_seed = 2 ** 32 - 1 - seed if test else seed
#     env.seed(int(env_seed))
#     # Cast observations to float32 because our model uses float32
#     env = chainerrl.wrappers.CastObservationToFloat32(env)
#     # Normalize action space to [-1, 1]^n
#     env = chainerrl.wrappers.NormalizeActionSpace(env)
#     if args.monitor:
#         env = chainerrl.wrappers.Monitor(
#             env, args.outdir, force=True, video_callable=lambda _: True)
#     if args.render:
#         env = chainerrl.wrappers.Render(env, mode='human')
#     return env


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--vrep-dir', type=str, required=True,
                        help='Directory path to vrep')
    parser.add_argument('--sim-path', type=str, required=True,
                        help='File path to simulation.ttt file')
    parser.add_argument('--env', type=str,
                        default='RoboschoolAtlasForwardWalk-v1',
                        help='OpenAI Gym env to perform algorithm on.')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of envs run in parallel.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--load', type=str, default='',
                        help='Directory to load agent from.')
    parser.add_argument('--steps', type=int, default=50000,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--eval-n-runs', type=int, default=20,
                        help='Number of episodes run for each evaluation.')
    parser.add_argument('--eval-interval', type=int, default=2500,
                        help='Interval in timesteps between evaluations.')
    parser.add_argument('--replay-start-size', type=int, default=250,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--update-interval', type=int, default=1,
                        help='Interval in timesteps between model updates.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Minibatch size')
    parser.add_argument('--render', action='store_true',
                        help='Render env states in a GUI window.')
    parser.add_argument('--demo', action='store_true',
                        help='Just run evaluation, not training.')
    parser.add_argument('--monitor', action='store_true',
                        help='Wrap env with Monitor to write videos.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Interval in timesteps between outputting log'
                             ' messages during training')
    parser.add_argument('--logger-level', type=int, default=logging.INFO,
                        help='Level of the root logger.')
    # parser.add_argument('--n-hidden-channels', type=int, default=1024,
    #                     help='Number of hidden channels of NN models.')
    parser.add_argument('--n-hidden-channels', type=int, default=1000,
                        help='Number of hidden channels of NN models.')
    parser.add_argument('--discount', type=float, default=0.98,
                        help='Discount factor.')
    parser.add_argument('--n-step-return', type=int, default=3,
                        help='N-step return.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate.')
    parser.add_argument('--adam-eps', type=float, default=1e-1,
                        help='Adam eps.')
    parser.add_argument('--sleeptime-before-bootup', type=int, default=4,
                        help='sleeptime before a simulator boots up')
    parser.add_argument('--display', type=str, default=None,
                        help='set DISPLAY env (default: None)')
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)
    logging.getLogger('PhysIntuitionEnv').setLevel(args.logger_level)
    logging.getLogger('SoftActorCriticAgent').setLevel(logging.DEBUG)


    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    lock = Lock()
    def make_batch_env(test):
        return MultiprocessVectorEnv(
            [functools.partial(make_env, args, process_seeds[idx], idx, test, lock)
             for idx, env in enumerate(range(args.num_envs))])

    sample_env = make_env(args, process_seeds[0], -1, test=False)
    # timestep_limit = sample_env.spec.max_episode_steps
    timestep_limit = 20
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print('Observation space:', obs_space)
    print('Action space:', action_space)
    del sample_env
    time.sleep(6)

    action_size = action_space.low.size

    winit = chainer.initializers.GlorotUniform()
    winit_policy_output = chainer.initializers.GlorotUniform()

    def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == action_size * 2
        mean, log_scale = F.split_axis(x, 2, axis=1)
        log_scale = F.clip(log_scale, -20., 2.)
        var = F.exp(log_scale * 2)
        return chainerrl.distribution.SquashedGaussianDistribution(
            mean, var=var)

    policy = chainer.Sequential(
        L.Linear(None, args.n_hidden_channels, initialW=winit),
        F.relu,
        L.Linear(None, args.n_hidden_channels, initialW=winit),
        F.relu,
        L.Linear(None, action_size * 2, initialW=winit_policy_output),
        squashed_diagonal_gaussian_head,
    )
    policy_optimizer = optimizers.Adam(
        args.lr, eps=args.adam_eps).setup(policy)

    # def make_q_func_with_optimizer():
    #     q_func = chainer.Sequential(
    #         concat_obs_and_action,
    #         L.Linear(None, args.n_hidden_channels, initialW=winit),
    #         F.relu,
    #         L.Linear(None, args.n_hidden_channels, initialW=winit),
    #         F.relu,
    #         L.Linear(None, 1, initialW=winit),
    #     )
    #     q_func_optimizer = optimizers.Adam(
    #         args.lr, eps=args.adam_eps).setup(q_func)
    #     return q_func, q_func_optimizer

    def make_q_func_with_optimizer():
        q_func = my_vgg.MyVGG16Layers()
        q_func_optimizer = optimizers.Adam(
            args.lr, eps=args.adam_eps).setup(q_func)
        return q_func, q_func_optimizer

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()

    # Draw the computational graph and save it in the output directory.
    # fake_obs = chainer.Variable(
    #     policy.xp.zeros_like(obs_space.low, dtype=np.float32)[None],
    #     name='observation')
    # fake_action = chainer.Variable(
    #     policy.xp.zeros_like(action_space.low, dtype=np.float32)[None],
    #     name='action')
    # chainerrl.misc.draw_computational_graph(
    #     [policy(fake_obs)], os.path.join(args.outdir, 'policy'))
    # chainerrl.misc.draw_computational_graph(
    #     [q_func1(fake_obs, fake_action)], os.path.join(args.outdir, 'q_func1'))
    # chainerrl.misc.draw_computational_graph(
    #     [q_func2(fake_obs, fake_action)], os.path.join(args.outdir, 'q_func2'))

    rbuf = replay_buffer.ReplayBuffer(10 ** 6, num_steps=args.n_step_return)

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(
            action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    import soft_actor_critic
    # agent = chainerrl.agents.SoftActorCritic(
    agent = soft_actor_critic.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=args.discount,
        update_interval=args.update_interval,
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        logger=logging.getLogger("SoftActorCriticAgent"),
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size,
        temperature_optimizer=chainer.optimizers.Adam(
            args.lr, eps=args.adam_eps),
    )


    if len(args.load) > 0:
        agent.load(args.load)

    print('num_envs:', args.num_envs)
    if args.demo:
        eval_env = make_env(args, seed=0, test=True)
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(test=False),
            eval_env=make_batch_env(test=True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            max_episode_len=timestep_limit,
        )


if __name__ == '__main__':
    main()
