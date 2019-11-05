import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
        sess,
        env,
        ac_dim,
        dyn_models,
        horizon,
        N,
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.sess = sess
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        # TODO(Q1) uniformly sample trajectories and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim)
        answer = np.array([self.ac_space.sample() for i in range(num_sequences * horizon)])
        return answer.reshape((num_sequences, horizon) + answer.shape[1:])

    def get_action(self, obs):

        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        #sample random actions (Nxhorizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon)

        # a list you can use for storing the predicted reward for each candidate sequence
        predicted_rewards_per_ens = [None]*len(self.dyn_models)

        for i, model in enumerate(self.dyn_models):
            # DONE(Q2)
            actions_sequence = np.swapaxes(candidate_action_sequences, 0, 1)
            observations = np.tile(obs, (self.N,1))
            states_sequence = model.helper(observations, actions_sequence, self.data_statistics)
            rewarded_states = np.vstack([np.expand_dims(observations, axis = 0), np.array(states_sequence[:-1])])
            rewards = [None]*self.horizon
            for j in range(self.horizon):
                rewards[j], done = self.env.get_reward(rewarded_states[j], actions_sequence[j])
            predicted_rewards_per_ens[i] = np.array(rewards).sum(axis=0)

            # for each candidate action sequence, predict a sequence of
            # states for each dynamics model in your ensemble

            # once you have a sequence of predicted states from each model in your
            # ensemble, calculate the reward for each sequence using self.env.get_reward (See files in envs to see how to call this)

        # calculate mean_across_ensembles(predicted rewards).
        # the matrix dimensions should change as follows: [ens,N] --> N
        predicted_rewards = np.mean(predicted_rewards_per_ens, axis=0) # TODO(Q2)

        # pick the action sequence and return the 1st element of that sequence
        best_index = np.argmax(predicted_rewards) #TODO(Q2)
        best_action_sequence = candidate_action_sequences[best_index] #TODO(Q2)
        action_to_take = best_action_sequence[0] # TODO(Q2)
        return action_to_take[None] # the None is for matching expected dimensions

    def get_action_helper(self, obs, ac, ds):
        return 
