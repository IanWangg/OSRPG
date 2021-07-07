"""
Example of running PyTorch implementation of DDPG on HalfCheetah.
"""
import sys
sys.path.append('../')
import copy
from rlkit.torch.ddpg.dsfpg import DSFPGTrainer
from gym.envs.mujoco import HopperEnv

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.networks import LinearMlp
from rlkit.torch.ddpg.ddpg import DDPGTrainer
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


import h5py
import d4rl, gym
import numpy as np

def load_hdf5(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)  
    replay_buffer._size = dataset['terminals'].shape[0]
    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size


def experiment(variant):
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env


    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    sf = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=obs_dim,
        **variant['sf_kwargs']
    )
    # qf can be either linear or non-linear function
    qf = LinearMlp(
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_sf = copy.deepcopy(sf)
    target_qf = copy.deepcopy(qf)
    target_policy = copy.deepcopy(policy)
    eval_path_collector = MdpPathCollector(eval_env, policy)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=OUStrategy(action_space=expl_env.action_space),
        policy=policy,
    )
    expl_path_collector = MdpPathCollector(expl_env, exploration_policy)
    # replay_buffer = EnvReplayBuffer(variant['replay_buffer_size'], expl_env)

    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']
    
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    if variant['load_buffer'] and buffer_filename is not None:
        replay_buffer.load_buffer(buffer_filename)
    else:
        load_hdf5(d4rl.qlearning_dataset(eval_env), replay_buffer)

    # need to change here 
    trainer = DSFPGTrainer(
        sf=sf,
        target_sf=target_sf,
        qf=qf,
        target_qf=target_qf,
        policy=policy,
        target_policy=target_policy,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        offline=variant['load_buffer'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm='OSFPG',
        version='normal',
        env_name='hopper-random-v0',
        load_buffer=True, # True value makes the agent trained under offline setting
        buffer_filename=None,
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=10000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            use_soft_update=True,
            tau=5e-3,
            discount=0.99,
            qf_learning_rate=1e-4,
            sf_learning_rate=2e-4,
            policy_learning_rate=1e-4,
        ),
        sf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        replay_buffer_size=int(1E6),
    )



    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    setup_logger('Test OSFPG offline in hopper env ---', variant=variant)
    experiment(variant)

    
