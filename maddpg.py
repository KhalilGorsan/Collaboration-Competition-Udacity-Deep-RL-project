import numpy as np
import torch

from ddpg import Agent


class Maddpg:
    def __init__(self, random_seed, state_size, action_size, replay_buffer, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed
        self.num_agents = num_agents
        self.shared_buffer = replay_buffer
        self.agents = [
            Agent(state_size, action_size, random_seed) for _ in range(num_agents)
        ]

    def step(self, states, actions, rewards, next_states, dones):
        self.shared_buffer.add(states, actions, rewards, next_states, dones)

        for agent in self.agents:
            agent.step()

    def act(self, states, add_noise=True):
        actions = np.zeros([self.num_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index], add_noise)
        return actions

    def save_weights(self):
        for index, agent in enumerate(self.agents):
            torch.save(
                agent.actor_local.state_dict(),
                "agent{}_checkpoint_actor.pth".format(index + 1),
            )
            torch.save(
                agent.critic_local.state_dict(),
                "agent{}_checkpoint_critic.pth".format(index + 1),
            )

    def reset(self):
        for agent in self.agents:
            agent.reset()
