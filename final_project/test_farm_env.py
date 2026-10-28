"""Smoke test: API check, baseline comparison, and a speed measurement."""
import time
import numpy as np
from gymnasium.utils.env_checker import check_env
from farm_env import FarmEnv, CROPS


def run(env, policy, seed):
    env.reset(seed=seed)
    done = False
    while not done:
        _, _, done, _, info = env.step(policy(env))
    return info["gold"]


def random_policy(env):
    return env.action_space.sample()


def scripted_policy(env):
    """Tends only the tile it starts on: till, plant, water, wait, harvest."""
    x, y = env.pos
    i = env.crop[x, y]
    if not env.tilled[x, y]:
        return env.A_TILL
    if i < 0:
        left = env.max_days - env.day
        if left >= 9:
            return env.A_PLANT + 1  # pumpkin
        if left >= 6:
            return env.A_PLANT      # radish
        return env.A_SLEEP
    if env.growth[x, y] >= CROPS[i][1]:
        return env.A_HARVEST
    if not env.watered[x, y] and not env.rain_tonight:
        return env.A_WATER
    return env.A_SLEEP


if __name__ == "__main__":
    env = FarmEnv()
    check_env(env)  # raises if the Gymnasium API is violated
    print("check_env passed")

    for name, pol in [("random", random_policy), ("scripted", scripted_policy)]:
        golds = [run(env, pol, s) for s in range(20)]
        print(f"{name:9s} final gold: mean {np.mean(golds):.1f} (std {np.std(golds):.1f})")

    env.reset(seed=0)
    t0, steps = time.time(), 0
    while time.time() - t0 < 2:
        _, _, done, _, _ = env.step(env.action_space.sample())
        steps += 1
        if done:
            env.reset()
    print(f"speed: {steps / 2:.0f} steps/sec (one process)")

    env.reset(seed=1)
    print(env.render())
