import argparse
from datetime import datetime
from pathlib import Path

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb

from zombie_battle_env import build_env


parser = argparse.ArgumentParser()
parser.add_argument('--total_timesteps', type=int, default=80000)
parser.add_argument('--save_freq', type=int, default=5000)
parser.add_argument('--msec_per_tick', type=int, default=20)
args = parser.parse_args()

work_dir = Path(__file__).resolve().parent
exp_name = 'exp_' + datetime.now().strftime('%Y-%m%d-%H%M%S')


def track_exp(project_name=None):
    wandb.init(
        anonymous='allow',
        dir=str(work_dir),
        project=project_name,
        sync_tensorboard=True,
        name=exp_name,
        monitor_gym=True,
        save_code=True,
    )


def make_env():
    def _build_env():
        env = build_env(msec_per_tick=args.msec_per_tick)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return _build_env


def train():
    exp_dir = work_dir / 'experiments' / exp_name
    tb_logs_dir = str(exp_dir / 'logs')
    weights_dir = str(exp_dir / 'model')

    env = DummyVecEnv([make_env()])
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=tb_logs_dir)
    callback = CheckpointCallback(save_freq=args.save_freq, save_path=weights_dir, name_prefix='model')

    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    model.save(weights_dir + '/model_final')
    env.close()


if __name__ == '__main__':
    track_exp(project_name=work_dir.name)
    train()
