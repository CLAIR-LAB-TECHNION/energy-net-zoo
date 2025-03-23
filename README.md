# energy-net-zoo

Integration of Energy-Net environments with RL-Zoo3 for training reinforcement learning agents.

## Getting Started

### Prerequisites

Make sure you have all the required packages installed:

```bash
pip install -r requirements.txt
```

### Training an Agent

You can train an agent with Energy-Net environments using the standard RL-Zoo3 commands:

```bash
python -m rl_zoo3.train --algo ppo --env PCSUnitEnv-v0 --env-kwargs demand_pattern:\"SINUSOIDAL\" cost_type:\"CONSTANT\"
```

Available algorithms:
- `ppo` - Proximal Policy Optimization
- `sac` - Soft Actor-Critic
- `td3` - Twin Delayed DDPG

Energy-Net parameters can be customized using the `--env-kwargs` argument:
- `demand_pattern`: `"CONSTANT"`, `"SINUSOIDAL"`, `"TIME_OF_DAY"`, etc.
- `cost_type`: `"CONSTANT"`, `"LMP"`, etc.
- Additional environment configuration options are available

### Evaluating a Trained Agent

After training, you can evaluate your agent using:

```bash
python -m rl_zoo3.enjoy --algo ppo --env PCSUnitEnv-v0 --env-kwargs demand_pattern:\"SINUSOIDAL\" cost_type:\"CONSTANT\" --no-render --n-episodes 10
```

## Advanced Usage

### Hyperparameter Optimization

RL-Zoo3 supports hyperparameter optimization using Optuna:

```bash
python -m rl_zoo3.train --algo ppo --env PCSUnitEnv-v0 --env-kwargs demand_pattern:\"SINUSOIDAL\" cost_type:\"CONSTANT\" --optimize --n-trials 100 --n-jobs 8 --sampler tpe
```

### Parallel Training

You can specify the number of parallel environments for training:

```bash
python -m rl_zoo3.train --algo ppo --env PCSUnitEnv-v0 --env-kwargs demand_pattern:\"SINUSOIDAL\" cost_type:\"CONSTANT\" --n-envs 8
```

### Plotting Results

After training, you can plot learning curves:

```bash
python -m rl_zoo3.plots.plot_train PCSUnitEnv-v0 --algo ppo sac td3
```

## Environment Configuration

The Energy-Net environment hyperparameters are defined directly in the standard RL-Zoo3 format in the following files:
- `hyperparams/ppo.yml`
- `hyperparams/sac.yml`
- `hyperparams/td3.yml`

For more detailed information on the Energy-Net environments and their configurations, refer to the `src/energy-net/README_RL_ZOO.md`.