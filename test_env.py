import gymnasium as gym
import numpy as np
from energy_net.market.iso.demand_patterns import DemandPattern
from energy_net.market.iso.cost_types import CostType
import matplotlib.pyplot as plt
from collections import defaultdict

# Register the environment
from energy_net.env.register_envs import register

def run_max_charging_agent(env, num_episodes=5, steps_per_episode=100):
    all_data = defaultdict(list)
    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        # Store initial observation (numpy array)
        all_data['obs_soc'].append(obs[0])  # State of Charge
        all_data['obs_time'].append(obs[1])  # Time
        all_data['obs_demand'].append(obs[2])  # Demand
        all_data['obs_price'].append(obs[3])  # Price
        
        for step in range(steps_per_episode):
            # Always charge at maximum rate (10.0 from the action space)
            action = np.array([10.0], dtype=np.float32)
            all_data['actions'].append(action[0])  # Store scalar action
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Store observations
            all_data['obs_soc'].append(obs[0])
            all_data['obs_time'].append(obs[1])
            all_data['obs_demand'].append(obs[2])
            all_data['obs_price'].append(obs[3])
            
            all_data['rewards'].append(reward)
            episode_reward += reward
            
            # Store additional info if available
            if isinstance(info, dict):
                for key, value in info.items():
                    all_data[f"info_{key}"].append(value)
            
            if terminated or truncated:
                break
                
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1} completed with total reward: {episode_reward:.2f}")
    
    return all_data, episode_rewards

def plot_results(all_data, episode_rewards):
    # Create figure with 2 subplots
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Combined SoC and Action
    ax2 = ax1.twinx()
    
    # Plot State of Charge with discrete points
    soc_plot = ax1.scatter(range(len(all_data['obs_soc'])), all_data['obs_soc'], 
                          color='blue', alpha=0.5, s=30, label='State of Charge')
    ax1.set_ylabel('State of Charge (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot Actions with discrete points
    action_plot = ax2.scatter(range(len(all_data['actions'])), all_data['actions'], 
                            color='red', alpha=0.5, s=30, label='Action (Charge/Discharge)')
    ax2.set_ylabel('Power (MW)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title('Battery State of Charge and Actions Over Time')
    ax1.set_xlabel('Step')
    
    # Add legends for first plot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot 2: Price over time
    ax3.scatter(range(len(all_data['obs_price'])), all_data['obs_price'], 
                color='green', alpha=0.5, s=30, label='Price')
    ax3.set_title('Electricity Price Over Time')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Price ($/MWh)')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Analyze and print the electricity price pattern
    time_points = all_data['obs_time']
    price_points = all_data['obs_price']
    
    # Find the pattern by looking at a single episode (100 steps)
    single_episode_prices = price_points[:100]
    single_episode_times = time_points[:100]
    
    # Calculate min, max, and average prices
    min_price = min(single_episode_prices)
    max_price = max(single_episode_prices)
    avg_price = sum(single_episode_prices) / len(single_episode_prices)
    
    print("\nElectricity Price Analysis:")
    print(f"Minimum Price: ${min_price:.2f}/MWh")
    print(f"Maximum Price: ${max_price:.2f}/MWh")
    print(f"Average Price: ${avg_price:.2f}/MWh")

def main():
    # Create the environment with fixed ISO (using CONSTANT cost type)
    env = gym.make(
        'PCSUnitEnv-v0',
        demand_pattern='CONSTANT',  # Fixed constant demand pattern
        cost_type='CONSTANT',       # Fixed constant cost type
        env_config_path='configs/environment_config.yaml',
        iso_config_path='configs/iso_config.yaml',
        pcs_unit_config_path='configs/pcs_unit_config.yaml'
    )
    
    print("\nEnvironment created successfully!")
    print("\nObservation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    
    # Run max charging agent and collect data
    all_data, episode_rewards = run_max_charging_agent(env, num_episodes=5, steps_per_episode=100)
    
    # Plot the results
    plot_results(all_data, episode_rewards)
    
    env.close()
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    main() 