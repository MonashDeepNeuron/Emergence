import argparse
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from env import RoverEnv

RUNS = Path(__file__).parent / "runs"
N_ENVS = 8

def train(timesteps):
    check_env(RoverEnv())   # catches obs/action space mistakes before you waste a run

    env = make_vec_env(RoverEnv, n_envs=N_ENVS, seed=0)
    eval_env = make_vec_env(RoverEnv, n_envs=1, seed=123)

    eval_cb = EvalCallback( # similar to K fold? the env for testing periodically tests the policy and saves the best version
        eval_env,
        best_model_save_path = str(RUNS / "best"),
        log_path = str(RUNS / "eval"),
        eval_freq = 50000 // N_ENVS, # how often to test the model. per 
        n_eval_episodes = 10, # how many episode each test run
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy", env,
        n_steps=1024, batch_size=256,
        gamma=0.995,
        ent_coef=0.03,  # early explore. award when random choose action
        verbose = 1, 
        seed = 0,
    )
    model.learn(total_timesteps=timesteps, callback=eval_cb)
    model.save(RUNS / "ppo_rover")


def test(path, episodes):
    env = RoverEnv()
    model = PPO.load(path)
    n_tiles = len(env.tiles)
    rewards, filled, lengths = [], [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            done = terminated or truncated
        rewards.append(total)
        filled.append(info["filled_tiles"])
        lengths.append(env.steps)
        print(f"episode {ep + 1:2d}: reward {total:7.2f}   "
              f"tiles {info['filled_tiles']:2d}/{n_tiles}   steps {env.steps}")

    filled = np.array(filled)
    print(f"\nmean reward      {np.mean(rewards):.2f}")
    print(f"mean tiles       {filled.mean():.1f} / {n_tiles}")
    print(f"pad completed    {np.mean(filled == n_tiles) * 100:.0f}% of episodes")
    print(f"mean steps       {np.mean(lengths):.0f}")


def watch(path):
    env = RoverEnv()
    model = PPO.load(path)
    obs, _ = env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            viewer.sync()
            time.sleep(0.05)
            if terminated or truncated:
                print(f"episode over: {info['filled_tiles']} tiles filled")
                obs, _ = env.reset()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--model", default=str(RUNS / "best" / "best_model.zip"))
    args = p.parse_args()

    if args.test:
        test(args.model, args.episodes)
    elif args.watch:
        watch(args.model)
    else:
        train(args.steps)


"""
  python train.py
  ../.venv/bin/mjpython train.py --watch
"""