
import numpy as np
from trainer import GeneralTrainer
import scipy.signal
from runningstat import RunningStats
from trpo import TRPO

class TRPOTrainer(GeneralTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.local_brain = TRPO(**kwargs)

        ''' 
        Running Statistics.
        normalize observations using running mean and std over the course of the entire experiment,
        fix the running statistics per batch
        see p.12 in https://arxiv.org/pdf/1707.02286.pdf
        '''
        self.running_stats = RunningStats(self.local_brain.env.get_state_shape()[0])


    ''' 
    core training routine. 
        updates value using previous batch of trajectories, 
        updates policy using current batch of trajectories,
        For details, see https://arxiv.org/pdf/1703.02660.pdf
    '''
    def train(self, session):
        self._print_instance_info()

        with session.as_default(), session.graph.as_default():
            self.intialize_params(session = session, n_episodes = 3)

            raw_t = self.gen_trajectories(session, self.local_brain.traj_batch_size)
            t_processed = self.process_trajectories(session,raw_t)
            self.update_policy(session, t_processed)
            t_processed_prev = t_processed

            while self.episode_count < self.max_episode_count:
                raw_t = self.gen_trajectories(session, self.local_brain.traj_batch_size)
                t_processed = self.process_trajectories(session, raw_t)

                self.update_policy(session, t_processed)
                self.update_value(t_processed_prev)

                self.auditor.log()
                t_processed_prev = t_processed


    ''' log, print run instance info. and hyper-params '''
    def _print_instance_info(self):
        self.auditor.update({'task': self.environ_string,
                          'seed': self.seed,
                          'max_episode_count': self.max_episode_count,
                          'policy_type': self.local_brain.policy_type,
                          'reward_discount': self.local_brain.reward_discount,
                          'gae_discount': self.local_brain.gae_discount,
                          'traj_batch_size': self.local_brain.traj_batch_size,
                          'n_policy_epochs': self.local_brain.n_policy_epochs,
                          'policy_learning_rate': float("%.5f" % self.local_brain.policy_learning_rate),
                          'value_learning_rate': float("%.5f" % self.local_brain.value_learning_rate),
                          'n_value_epochs': self.local_brain.n_value_epochs,
                          'value_batch_size': self.local_brain.value_batch_size,
                          'kl_target': self.local_brain.kl_target,
                          'beta': self.local_brain.beta,
                          'beta_min': self.local_brain.beta_min,
                          'beta_max': self.local_brain.beta_max,
                          'ksi': self.local_brain.ksi
                          })
        self.auditor.logmeta()

        return self


    ''' Initialize environment dependent parameters, such as running mean + std '''
    def intialize_params(self, session, n_episodes):
        self.gen_trajectories(session, n_episodes)
        return self


    ''' generate a single episodic trajectory '''
    def _gen_trajectory(self, session):
        state = self.local_brain.env.reset_environment()
        actions, rewards, states, norm_states = [], [], [], []
        terminal = False

        while terminal is False:

            states.append(state)

            state_normalized = (state - self.running_stats.mean()) / self.running_stats.standard_deviation()
            norm_states.append(state_normalized)

            action = self.local_brain.sample_action(session, state_normalized)
            new_state, reward, terminal, info = self.env.perform_action(action)
            actions.append(action)
            rewards.append(reward)

            state = new_state # recurse and repeat until episode terminates
        return actions, rewards, states, norm_states


    def _discount(self, x, gamma):
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


    ''' generate trajectories by rolling out the stochastic policy 'pi_theta_k', of iteration k,
    and no truncation of rolling horizon, unless needed'''
    def gen_trajectories(self, session, traj_batch_size):

        raw_t = {'states':[], 'actions':[], 'rewards':[], 'disc_rewards':[], 'values':[], 'advantages':[]}
        raw_states = []
        for episode in range(traj_batch_size):

            actions, rewards, states, norm_states = self._gen_trajectory(session)

            raw_t['states'].append(norm_states)
            raw_t['actions'].append(actions)
            raw_t['rewards'].append(rewards)
            ''' discounted sum of rewards until the end of episode for value update'''
            raw_t['disc_rewards'].append(self._discount(rewards, gamma = self.local_brain.reward_discount))

            raw_states += states
            self.episode_count += 1

        self.running_stats.multiple_push(raw_states) # per batch update running statistics

        self.auditor.update({'episode_number': self.episode_count,
                          'per_episode_mean': int(np.sum(np.concatenate(raw_t['rewards'])) / traj_batch_size)
                          })

        return raw_t


    ''' estimate value and advantages: gae'''
    def process_trajectories(self, session, t):
        for i in range(self.local_brain.traj_batch_size):
            feed_dict = {self.local_brain.input_ph: t['states'][i]}
            values = session.run(self.local_brain.value, feed_dict=feed_dict)
            t['values'].append(values)

            ''' generalized advantage estimation from https://arxiv.org/pdf/1506.02438.pdf for policy gradient update'''
            temporal_differences = t['rewards'][i] + np.append(self.local_brain.reward_discount * values[1:], 0.0) - list(map(float, values))
            gae = self._discount(temporal_differences, self.local_brain.gae_discount)

            t['advantages'].append(gae)

        t['states'] = np.concatenate(t['states'])
        t['actions'] = np.concatenate(t['actions'])
        t['rewards'] = np.concatenate(t['rewards'])
        t['disc_rewards'] = np.concatenate(t['disc_rewards'])
        t['values'] = np.concatenate(t['values'])

        ''' per batch normliazation of gae. see p.13 in https://arxiv.org/pdf/1707.02286.pdf '''
        concatenated_gae = np.concatenate(t['advantages'])
        normalized_gae = (concatenated_gae - concatenated_gae.mean()) / (concatenated_gae.std() + 1e-6)
        t['advantages'] = normalized_gae

        t['actions'] = np.reshape(t['actions'], (-1, self.local_brain.env_action_number))
        for entity in ['rewards', 'disc_rewards', 'values', 'advantages']:
            t[entity] = np.reshape(t[entity], (-1, 1))

        return t

    ''' updates policy '''
    def update_policy(self, session, t):
        self.local_brain._update_policy(session, t, self.auditor)
        return self

    ''' updates value '''
    def update_value(self, t):
        self.local_brain._update_value(t, self.auditor)
        return self
