import time
import dmc2gym

if __name__ == '__main__':
    startTime = time.time()
    env = dmc2gym.make(
        domain_name='quadruped',
        task_name='run',
        seed=0,
        visualize_reward=False,
        from_pixels=True,
        height=100,
        width=100,
        frame_skip=4
    )
    n_episodes = 10
    for i in range(n_episodes):
        print(i)
        obs = env.reset()
        done = False
        t = 0
        while not done:
            t += 1
            # center crop image
            # if args.encoder_type == {'pixel', 'fourier_pixel', 'fair_pixel'} and 'translate' in args.data_augs:
                # obs = utils.center_crop_image(obs, args.pre_transform_image_size)
                # obs = utils.center_translate(obs, args.image_size)
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
        print('timesteps', t)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))