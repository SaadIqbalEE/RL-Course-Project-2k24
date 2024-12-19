from .agent_base import BaseAgent
from .ppo_utils import Policy
from .ppo_agent import PPOAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time
from collections import deque

class PPOExtension(PPOAgent):
    def __init__(self, config=None):
        super(PPOExtension, self).__init__(config)
        
        # Hit&trail Improvement
        sil_custom_buffer_size = 128#sil_buffer_size
        sil_custom_batch_size = 64#sil_batch_size
        sil_custom_epochs = 3#sil_epochs
        #sil_custom_coef = #sil_coef
        sil_custom_reward_threshold = 0#sil_reward_threshold

        self.sil_buffer = deque(maxlen=sil_custom_buffer_size)
        self.sil_batch_size = sil_custom_batch_size
        self.sil_epochs = sil_custom_epochs
        
        # Weighting factor for SIL loss  
        #self.sil_coef = sil_custom_coef
        self.sil_reward_threshold = sil_custom_reward_threshold

    def update_policy(self):
       #########ADDED###################
        if not self.silent:
            print("Updating the policy...")

        self.states = torch.stack(self.states)
        self.actions = torch.stack(self.actions).squeeze()
        self.next_states = torch.stack(self.next_states)
        self.rewards = torch.stack(self.rewards).squeeze()
        self.dones = torch.stack(self.dones).squeeze()
        self.action_log_probs = torch.stack(self.action_log_probs).squeeze()

        for e in range(self.epochs):
            self.ppo_epoch()
            
        # Perform SIL update
        for e in range(self.sil_epochs):
            self.sil_update()
            
        # Clear the replay buffer
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.action_log_probs = []
        if not self.silent:
            print("Updating finished!")

    def sil_update(self):
        if len(self.sil_buffer) < self.sil_batch_size:
            return  # Skip if buffer is insufficient
        indices = np.arange(len(self.sil_buffer))  # Create indices for buffer elements
        sampled_indices = np.random.choice(indices, self.sil_batch_size, replace=False)
        batch = [self.sil_buffer[i] for i in sampled_indices]  # Retrieve sampled elements
        states, actions, rewards, dones = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)

        # Compute SIL loss
        with torch.no_grad():
            _, target_values = self.policy(states)
        target_values = target_values.squeeze()
        advantages = rewards - target_values
        
        # Imitate only if reward > value
        advantages = advantages.clamp(min=0)  
        
        action_dists, values = self.policy(states)
        log_probs = action_dists.log_prob(actions).sum(-1)

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.smooth_l1_loss(values.squeeze(), rewards)
        
        loss = policy_loss + 0.5 * value_loss

        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_iteration(self,ratio_of_episodes):
        # Run actual training        
        reward_sum, episode_length, num_updates = 0, 0, 0
        done = False

        # Reset the environment and observe the initial state
        observation, _  = self.env.reset()

        while not done and episode_length < self.cfg.max_episode_steps:
            # Get action from the agent
            action, action_log_prob = self.get_action(observation)
            previous_observation = observation.copy()

            # Perform the action on the environment, get new state and reward
            observation, reward, done, _, _ = self.env.step(action)
            
            # Store action's outcome (so that the agent can improve its policy)
            self.store_outcome(previous_observation, action, observation,
                                reward, action_log_prob, done)

            # Store total episode reward
            reward_sum += reward
            episode_length += 1

            # Add experience to SIL buffer
            if reward > self.sil_reward_threshold:
                self.sil_buffer.append((
                    torch.from_numpy(previous_observation).float(),
                    torch.Tensor(action).float(),
                    torch.Tensor([reward]).float(),
                    torch.Tensor([done])
                ))            

            # Update the policy, if we have enough data
            if len(self.states) > self.cfg.min_update_samples:
                self.update_policy()
                num_updates += 1

                # Update policy randomness
                self.policy.set_logstd_ratio(ratio_of_episodes)

        # Return stats of training
        update_info = {'episode_length': episode_length,
                    'ep_reward': reward_sum}
        return update_info