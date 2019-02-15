import torch


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size, collect_traj=False):
        self.obs_shape = obs_shape
        self.state_size = state_size
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.states = torch.zeros(num_steps + 1, num_processes, state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            self.action_shape = 1
        else:
            self.action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, self.action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0
        self.collect_traj = collect_traj

        # if collecting trajectories
        if collect_traj:
            self.observations_orig = torch.zeros(num_steps + 1, num_processes, *obs_shape)
            self.size_db = 0
            self.db_index = 0
            self.observations_db = torch.zeros(num_steps + 1, num_processes, *obs_shape)
            self.states_db = torch.zeros(num_steps + 1, num_processes, state_size)
            self.actions_db = torch.zeros(num_steps, num_processes, self.action_shape)
            self.returns_db = torch.zeros(num_steps + 1, num_processes, 1)
            self.max_db_size = num_steps

    def get_item_from_db(self, i, batch_size):
        """
        :param i: index in db (buffer)
        :param batch_size: how many samples to return
        :return: demonstrations from the buffer
        """
        return self.observations_db[i:i + batch_size].squeeze(1), self.actions_db[i:i + batch_size].squeeze(1), \
               self.returns_db[i:i + batch_size].squeeze(1)

    def copy_to_db(self):
        """
        copy data from current buffer to db
        """
        if self.db_index + self.step > self.num_steps:
            self.db_index = 0
        else:
            self.size_db += self.step
        self.observations_db[self.db_index:self.db_index+self.step] = self.observations.narrow(0, 0, self.step).cpu().clone()
        self.states_db[self.db_index:self.db_index+self.step] = self.states.narrow(0, 0, self.step).cpu().clone()
        self.actions_db[self.db_index:self.db_index+self.step] = self.actions.narrow(0, 0, self.step).cpu().clone()
        self.returns_db[self.db_index:self.db_index+self.step] = self.returns.narrow(0, 0, self.step).cpu().clone()
        self.db_index += self.step

    def cuda(self):
        self.observations = self.observations.cuda()
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, current_obs, state, action, action_log_prob, value_pred, reward, mask, obs_orig=None):
        """
        insert data to the current buffer
        :param obs_orig: original observation (from the target task)
        """
        self.observations[self.step + 1].copy_(current_obs)
        if self.collect_traj:
            self.observations_orig[self.step + 1].copy_(obs_orig)
        self.states[self.step + 1].copy_(state)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self, collect=False):
        """
        initialize buffers
        :param collect: True for collecting trajectories, False otherwise
        """
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])
        if collect:
            self.step = 0

    def compute_returns(self, next_value, gamma, collect=False):
        """
        compute accumulated rewards for each state
        :param next_value: value of the next state
        :param gamma: value of the discount factor
        :param collect: True for collecting trajectories, False otherwise
        """
        if collect:
            self.returns = torch.zeros(self.step, 1, 1)
            rewards = self.rewards.narrow(0, 0, self.step)
            masks = self.masks.narrow(0, 0, self.step + 1)
            if self.rewards.is_cuda:
                self.returns = self.returns.cuda()
                rewards = rewards.cuda()
                masks = masks.cuda()
            self.returns[-1] = rewards[-1]
            rewards_len = rewards.size(0) - 1
        else:
            rewards = self.rewards
            masks = self.masks
            self.returns[-1] = next_value
            rewards_len = rewards.size(0)
        for step in reversed(range(rewards_len)):
            self.returns[step] = self.returns[step + 1] * \
                gamma * masks[step + 1] + rewards[step]
