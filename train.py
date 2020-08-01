# main function that sets up environments
# perform training loop

import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

from buffer import ReplayBuffer
from core import TennisWrapper
from maddpg import MADDPG
from utilities import transpose_list, transpose_to_tensor


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    seeding()
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 30000
    episode_length = 80
    batchsize = 1000
    # how many episodes to save policy and gif
    save_interval = 1000

    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 2
    noise_reduction = 0.9999

    # how many episodes before update
    episode_per_update = 2

    log_path = os.getcwd() + "/log"
    model_dir = os.getcwd() + "/model_dir"

    os.makedirs(model_dir, exist_ok=True)

    # keep 5000 episodes worth of replay
    buffer = ReplayBuffer(int(5000 * episode_length))

    # initialize policy and critic
    maddpg = MADDPG()
    logger = SummaryWriter(log_dir=log_path)
    agent0_reward = []
    agent1_reward = []

    # instantiate environment object
    env = TennisWrapper(file_name="./Tennis")

    # training loop
    for episode in range(0, number_of_episodes):

        reward_this_episode = np.zeros((1, 3))
        all_obs = env.reset()  #
        obs, obs_full = transpose_list(all_obs)

        # for calculating rewards for this particular episode - addition of all time steps
        # save info or not
        save_info = (episode) % save_interval < 1 or episode == number_of_episodes - 1

        for _ in range(episode_length):

            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            actions = maddpg.act(transpose_to_tensor(obs), noise=noise)
            noise *= noise_reduction

            actions_array = torch.stack(actions).detach().numpy()

            # transpose the list of list
            # flip the first two indices
            # input to step requires the first index to correspond to number of parallel agents
            actions_for_env = np.rollaxis(actions_array, 1)

            # step forward one frame
            next_obs, next_obs_full, rewards, dones = env.step(actions_for_env)

            # add data to buffer
            transition = (
                obs,
                obs_full,
                actions_for_env,
                rewards,
                next_obs,
                next_obs_full,
                dones,
            )

            buffer.push(transition)

            reward_this_episode += rewards

            obs, obs_full = next_obs, next_obs_full

        # update once after every episode_per_update
        if len(buffer) > batchsize and episode % episode_per_update < 1:
            for a_i in range(2):
                samples = buffer.sample(batchsize)
                maddpg.update(samples, a_i, logger)
            maddpg.update_targets()  # soft update the target network towards the actual networks

        agent0_reward.append(reward_this_episode[0, 0])
        agent1_reward.append(reward_this_episode[0, 1])

        if episode % 100 == 0 or episode == number_of_episodes - 1:
            avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward)]
            agent0_reward = []
            agent1_reward = []

            for a_i, avg_rew in enumerate(avg_rewards):
                logger.add_scalar(
                    "agent%i/mean_episode_rewards" % a_i, avg_rew, episode
                )

        # saving model
        save_dict_list = []
        if save_info:
            for i in range(2):

                save_dict = {
                    "actor_params": maddpg.maddpg_agent[i].actor.state_dict(),
                    "actor_optim_params": maddpg.maddpg_agent[
                        i
                    ].actor_optimizer.state_dict(),
                    "critic_params": maddpg.maddpg_agent[i].critic.state_dict(),
                    "critic_optim_params": maddpg.maddpg_agent[
                        i
                    ].critic_optimizer.state_dict(),
                }
                save_dict_list.append(save_dict)

                torch.save(
                    save_dict_list,
                    os.path.join(model_dir, "episode-{}.pt".format(episode)),
                )

    env.close()
    logger.close()


if __name__ == "__main__":
    main()
