from setuptools import setup, find_packages

setup(
    name="energy_net_rl_zoo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.4",
        "scipy>=1.10.1",
        "gymnasium>=1.0.0",
        "pettingzoo>=1.24.3",
        "stable-baselines3>=2.4.1",
        "PyYAML>=6.0.2",
        "matplotlib>=3.7.5",
        "torch>=2.2.2",
        "sb3-contrib>=2.4.1",
        "optuna>=3.4.0",
        "tqdm>=4.66.2",
        "moviepy>=1.0.3",
    ],
    description="Energy-Net with RL-Zoo3 integration",
    author="Energy-Net Team",
    python_requires=">=3.7",
) 