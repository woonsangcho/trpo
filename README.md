# Proximal Policy Optimization Implementation using Tensorflow and Keras

Code written by Galen Cho (Woon Sang Cho): https://github.com/woonsangcho

### Summary

This is an implementation of Proximal Policy Optimization (PPO)[1][2], which is a variant of Trust Region Policy Optimization TRPO)[3].

This is one version that resulted from experimenting a number of variants, in particular with loss functions, advantages[4], normalization, and a few other tricks in the reference papers.  I have clearly commented so that it is easy to follow, along with page numbers wherever necessary.


### Implementation notes
- The parametrized value function is updated using the previous batch of trajectories, whereas the parametrized policy function and advantages are updated using the current batch of trajectories, as suggested in [5] to avoid overfitting. Please see the reference [5] for the core training scheme.
- The network architecture for both value and policy is fixed with ```two``` hidden layers, each with ```128``` nodes with ```tanh``` activation. 
- The kernels in the layers are initialized with ```RandomNormal(mean=0, std=0.1)```, which is a heuristic I've found to be useful. The effect is that the output for mean action are centered around 0 during the initial stage of policy learning. If this center a very much off from 0, resulting in large random fluctuation of actions, and taking longer time to learn.
- While it is noted in [3] that a ```separate set of parameters specifies the log standard deviation of each element```, I've experimented with merged network outputting both ```mean``` and ```sigma```, and with two separate networks for each. They had poor performance so the source is omitted from the repository. 
- The value function is trained using the built-in fit routine in keras for convenient epoch and batch-size management. The policy function is trained using tensorflow over the entire batch of trajectories. You may modify the source to truncate the episodic rollout to a fixed horizon size ```T```. The suggested size is ```T=2048```, noted in the reference.

### Dependencies
- [tensorflow](https://github.com/tensorflow/tensorflow) (1.4.0)
- [keras](https://github.com/keras-team/keras) (2.0.9)
- numpy
- scipy
- [openai gym](https://github.com/openai/gym) (0.9.4)
- [MuJoCo](https://github.com/openai/mujoco-py)

### Sample run results in tensorboard


### Hyper-parameters
These hyper-parameters follow the references, with some futher tuning. Please see find my comments for details.
```
policy_learning_rate = 1 * 1e-04
value_learning_rate = 1.5 * 1e-03
n_policy_epochs = 20
n_value_epochs = 15
value_batch_size = 32
kl_target = 0.003
beta = 1
beta_max = 20
beta_min = 1/20
ksi = 10
reward_discount = 0.995
gae_discount = 0.975
traj_batch_size = 20 # per batch number of episodes to collect for each training iteration
activation = 'tanh'
```

### Run commands
```
python main.py --environ-string='InvertedPendulum-v1' --max-episode-count=1000
python main.py --environ-string='InvertedDoublePendulum-v1' --max-episode-count=15000
python main.py --environ-string='Hopper-v1' --max-episode-count=20000
python main.py --environ-string='HalfCheetah-v1' --max-episode-count=15000
python main.py --environ-string='Swimmer-v1' --max-episode-count=5000
python main.py --environ-string='Ant-v1' --max-episode-count=150000
python main.py --environ-string='Reacher-v1' --max-episode-count=50000
python main.py --environ-string='Walker2d-v1' --max-episode-count=30000
python main.py --environ-string='Humanoid-v1' --max-episode-count=150000
python main.py --environ-string='HumanoidStandup-v1' --max-episode-count=150000
```
The default seed input is ```1989```. You can append ```--seed=<value>``` to experiment different seeds.





### References
1. [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) (Schulman et al. 2017)
2. [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf) (Heese et al. 2017)
3. [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) (Schulman et al. 2015)
4. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf) (Schulman et al. 2015)
5. [Towards Generalization and Simplicity in Continuous Control](https://arxiv.org/pdf/1703.02660.pdf) (Rajeswaran et al.)
6. [Repository 1 for helpful implementation pointers](https://github.com/joschu/modular_rl) (Schulman)
7. [Repository 2 for helpful implementation pointers](https://github.com/pat-coady/trpo) (Coady)
