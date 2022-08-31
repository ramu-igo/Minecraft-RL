import argparse

from stable_baselines3 import PPO

from zombie_battle_env import build_env


parser = argparse.ArgumentParser()
parser.add_argument('--trained_model', type=str, default=None)
parser.add_argument('--num_episode', type=int, default=5)
parser.add_argument('--msec_per_tick', type=int, default=50)
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()


def run_agent():
    env = build_env(msec_per_tick=args.msec_per_tick, interactive=args.interactive)

    if args.trained_model is None:
        model = PPO('CnnPolicy', env) # random action
    else:
        model = PPO.load(args.trained_model, env=env)

    for _ in range(args.num_episode):
        print('---------------------')
        obs = env.reset()

        steps = 0
        total_rewards = 0
        done = False
        while not done:
            action = model.predict(obs.copy())[0]
            obs, reward, done, info = env.step(action)
            steps += 1
            total_rewards += reward
            if reward != 0:
                print(f'reward: {reward:.2f}')

        print(f'Episode finished in {steps} steps with reward: {total_rewards:.2f} ')

    env.close()


if __name__ == '__main__':
    run_agent()
