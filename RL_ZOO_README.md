# Using RL-Zoo3 with Energy-Net

This repository includes the RL-Zoo3 integration for training reinforcement learning agents with the Energy-Net environment.

## Getting Started

### Prerequisites

Make sure you have all the required packages installed:

```bash
pip install -r requirements.txt
```

### Training an Agent

You can train an agent using the standard RL-Zoo3 CLI.

```bash
python -m rl_zoo3.train --algo ppo --env PCSUnitEnv-v0 --eval-freq 10000 --eval-episodes 5 -n 1000000 --env-kwargs demand_pattern:\"SINUSOIDAL\" cost_type:\"CONSTANT\"
```

### Evaluating a Trained Agent

```bash
python -m rl_zoo3.enjoy --algo ppo --env PCSUnitEnv-v0 --n-episodes 10 --env-kwargs demand_pattern:\"SINUSOIDAL\" cost_type:\"CONSTANT\"
```

### Custom Environment Parameters

You can customize the environment parameters when training with RL-Zoo3 using the `--env-kwargs` argument:

```bash
python -m rl_zoo3.train --algo ppo --env PCSUnitEnv-v0 --env-kwargs demand_pattern:\"TIME_OF_DAY\" cost_type:\"LMP\"
```

Available demand patterns: `CONSTANT`, `TIME_OF_DAY`, `SINUSOIDAL`, etc.
Available cost types: `CONSTANT`, `LMP`, etc.

## Advanced Usage

### Hyperparameter Optimization

RL-Zoo3 supports hyperparameter optimization using Optuna:

```bash
python -m rl_zoo3.train --algo ppo --env PCSUnitEnv-v0 --env-kwargs demand_pattern:\"SINUSOIDAL\" cost_type:\"CONSTANT\" --optimize --n-trials 100 --n-jobs 8 --sampler tpe
```

### Parallel Training

You can run multiple training runs in parallel:

```bash
python -m rl_zoo3.train --algo ppo --env PCSUnitEnv-v0 -n 1000000 --num-threads 4 --env-kwargs demand_pattern:\"SINUSOIDAL\" cost_type:\"CONSTANT\"
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