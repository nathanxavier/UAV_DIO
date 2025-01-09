from CoppeliaEnv.drone_env import DroneEnv
from gym.envs.registration import register

register(
    id = "CoppeliaEnv/Drone-v0",
    entry_point = "CoppeliaEnv:DroneEnv",
    )