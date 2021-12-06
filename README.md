# Reinforcement Learning with Learned Fourier Features


## State-space Soft Actor-Critic Experiments
Move to the `state-SAC-LFF` repository.
```
cd state-SAC-LFF
```
To install the dependencies, use the provided `environment.yml` file
```bash
conda env create -f environment.yml
```

To run an experiment, the template for MLP and LFF experiments, respectively, are:  
```bash
python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 3
python main.py --policy PytorchSAC --env dm.quadruped.run --start_timesteps 5000 --hidden_dim 1024 --batch_size 1024 --n_hidden 2 \
               --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --concatenate_fourier
```
The only thing that changes between the baseline is the number of hidden layers (we reduce by 1 to keep parameter count 
roughly the same), the `network_class`, the `fourier_dim`, `sigma`, `train_B`, and `concatenate_fourier`. 


## Image-space Soft Actor-Critic Experiments 
Move to the `image-SAC-LFF` repository. 
```bash
cd image-SAC-LFF
```

Install RAD dependencies: 
```bash
conda env create -f conda_env.yml
```

To run an experiment, the template for CNN and CNN+LFF experiments, respectively, are:  
```bash
python train.py --domain_name hopper --task_name hop --encoder_type fourier_pixel --action_repeat 4 \
                --num_eval_episodes 10 \--pre_transform_image_size 100 --image_size 84 --agent rad_sac \
                --frame_stack 3 --data_augs crop --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 \
                --num_train_steps 1000000 --fourier_dim 128 --sigma 0.1 --train_B --concatenate_fourier
python train.py --domain_name hopper --task_name hop --encoder_type fair_pixel --action_repeat 4 \
                --num_eval_episodes 10 \--pre_transform_image_size 100 --image_size 84 --agent rad_sac \
                --frame_stack 3 --data_augs crop --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 \
                --num_train_steps 1000000
```

## Proximal Policy Optimization Experiments 
Move to the `state-PPO-LFF` repository. 
```bash
cd pytorch-a2c-ppo-acktr-gail
```

Install PPO dependencies: 
```bash
conda env create -f environment.yml
```

To run an experiment, the template for MLP and LFF experiments, respectively, are:
```bash
python main.py --env-name Hopper-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 \
               --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 \
               --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits \
               --hidden_dim 256 --network_class MLP --n_hidden 2 --seed 10
python main.py --env-name Hopper-v2 --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 \
               --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 \
               --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits \
               --hidden_dim 256 --network_class FourierMLP --n_hidden 2 --sigma 0.01 --fourier_dim 64 \ 
               --concatenate_fourier --train_B --seed 10
```

## Acknowledgements
We built the state-based SAC codebase off the [TD3 repo by Fujimoto et al](https://github.com/sfujim/TD3). We especially 
appreciated its lightweight bare-bones training loop. For the state-based SAC algorithm implementation and hyperparameters, we used 
[this PyTorch SAC repo](https://github.com/denisyarats/pytorch_sac) by Yarats and Kostrikov. For the SAC+RAD image-based experiments, 
we used the authors' [implementation](https://github.com/MishaLaskin/rad). Finally, we built off 
[this PPO codebase](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) by Ilya Kostrikov. 