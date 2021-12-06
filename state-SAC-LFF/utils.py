import gym
import torch
import numpy as np
import dmc2gym

try:
    import metaworld
except ImportError:
    pass


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


def make_env(env_name, seed=10):
    if env_name.startswith("dm"):
        _, domain, task = env_name.split('.')
        env = dmc2gym.make(domain_name=domain,
                           task_name=task,
                           seed=seed,
                           visualize_reward=True)
        assert env.action_space.low.min() >= -1
        assert env.action_space.high.max() <= 1
    elif env_name.startswith('mw'):
        _, task = env_name.split('.')
        mt10 = metaworld.MT10()
        env = mt10.train_classes[task]()
        tasks = [t for t in mt10.train_tasks if t.env_name == task]
        env.set_task(tasks[0])
        if env.max_path_length <= 150:
            raise RuntimeError
        env._max_episode_steps = 150  # env.max_path_length
    else:
        env = gym.make(env_name)
    return env
