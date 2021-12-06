import json
import torch
import argparse
import TD3
import models.mlp
import numpy as np
from utils import make_env
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 0


# A fixed seed is used for the eval environment
def get_rollout_states(policy, env_name, noise=0, seed=0, timesteps=10000):
    print('Collecting timesteps')
    eval_env = make_env(env_name)
    eval_env.seed(seed + 100)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    states = []
    while len(states) < timesteps:
        print(f'### Collected {len(states)} timesteps so far ###')
        state, done = eval_env.reset(), False
        states.append(state)
        t = 0
        while t < env._max_episode_steps and not done:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * noise, size=action_dim)
            ).clip(-max_action, max_action)
            state, reward, done, _ = eval_env.step(action)
            states.append(state)
            t += 1
    return np.array(states)[:timesteps]


def save_embedding(states, fpath, embedding='pca'):
    if embedding == 'pca':
        x_embedded = PCA(n_components=2).fit_transform(states)
    elif embedding == 'tsne':
        x_embedded = TSNE(n_components=2, verbose=1).fit_transform(states)
    else:
        raise NotImplementedError
    plt.figure(dpi=200)
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=1)
    plt.title(f'{embedding} embedding of visited states')
    plt.savefig(fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--expl_noise", type=float, default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--n_timesteps", '-n', default=1e5, type=int)
    parser.add_argument("--load_model", default="")
    args = parser.parse_args()

    # set up, load
    env = make_env(args.env)
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }

    path = Path(args.load_model)
    folder = path.parent
    with open(folder / 'variant.json', "r") as read_file:
        variant = json.load(read_file)

    # custom network kwargs
    kwargs['network_class'] = eval(variant['network_class']['$class'])
    if 'Fourier' in variant['network_class']['$class']:
        kwargs['network_kwargs'] = variant['network_kwargs']
    else:
        kwargs['network_kwargs'] = dict(n_hidden=variant['network_kwargs']['n_hidden'],
                                        hidden_dim=variant['network_kwargs']['hidden_dim'],
                                        first_dim=variant['first_dim'])

    # Initialize policy
    if args.policy == "TD3":
        policy = TD3.TD3(**kwargs)
    else:
        raise NotImplementedError
    policy.load(args.load_model)

    states = get_rollout_states(policy, args.env, noise=args.expl_noise, seed=SEED, timesteps=args.n_timesteps)
    print('Saving states')
    np.save(folder / f'{path.name}_noise{args.expl_noise}_states.npy', states)
    print('Computing PCA embedding')
    save_embedding(states, folder / f'{path.name}_noise{args.expl_noise}_pca.png', embedding='pca')
    print('Computing tSNE embedding')
    save_embedding(states, folder / f'{path.name}_noise{args.expl_noise}_tsne.png', embedding='tsne')
