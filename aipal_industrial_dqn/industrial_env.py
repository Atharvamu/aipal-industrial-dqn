import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class IndustrialAutomationEnv(gym.Env):
    def __init__(self):     #Initialize the environment
        super().__init__()

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32) #5 observation spaces because we have 5 parameters to observe which are machine health, product quality, time since maintenance, param1, and param2
        self.action_space = spaces.Discrete(6)  # 0: no action, 1–2: param1 up/down, 3–4: param2 up/down, 5: maintenance

        self.metrics = {
            'healths': [],
            'qualities': [],
            'param1s': [],
            'param2s': [],
            'rewards': []
        }                                   #storing metrics for plotting later	
        self.fig, self.ax = None, None
        self.max_steps = 300 #300 episode length instead of 100 to allow more time for learning

        self.reset()

    def reset(self, *, seed=None, options=None): #Environment restart after each episode
        super().reset(seed=seed)

        np.random.seed(seed)
        self.machine_health = np.clip(np.random.normal(loc=0.9, scale=0.05), 0.7, 1.0) # Initial health is normally distributed around 0.9 with a small standard deviation
        self.product_quality = np.random.uniform(0.4, 0.7) #Random quiality between 0.4 and 0.7
        self.time_since_maintenance = np.random.choice([0.0, np.random.uniform(0.1, 0.3)], p=[0.7, 0.3]) #70% chance of starting with no maintenance, 30% chance of a small delay
        self.param1 = np.random.uniform(0.4, 0.6) # Random control parameter 1 between 0.4 and 0.6
        self.param2 = np.random.uniform(0.4, 0.6) # Random control parameter 2 between 0.4 and 0.6
        self.steps = 0

        for key in self.metrics: # Clear metrics log for new episode
            self.metrics[key] = []

        return self._get_obs(), {}

    def _get_obs(self):       #constructs the state vector seen by the agent
        return np.array([
            self.machine_health,
            self.product_quality,
            self.time_since_maintenance,
            self.param1,
            self.param2
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        self.time_since_maintenance += 0.05

        # Control parameter adjustment
        if action == 1: # Increase param1
            self.param1 = min(self.param1 + 0.1, 1.0)
        elif action == 2:
            self.param1 = max(self.param1 - 0.1, 0.0)
        elif action == 3:
            self.param2 = min(self.param2 + 0.1, 1.0)
        elif action == 4:
            self.param2 = max(self.param2 - 0.1, 0.0)
        elif action == 0:   #No action taken, but still update parameters slightly according tovnatural variation
            self.param1 = np.clip(np.random.normal(self.param1, 0.01), 0.0, 1.0)
            self.param2 = np.clip(np.random.normal(self.param2, 0.01), 0.0, 1.0)

        # Maintenance action (realistic logic)
        reward = 0.0
        if action == 5:
            # Effectiveness is sigmoid-shaped: early = wasteful, late = diminishing return
            effectiveness = 0.9 + 0.1 * np.tanh(5 * (self.time_since_maintenance - 0.25)) #How effective maintenance is;If maintenance is done too early (< 0.25), effectiveness is near 0.9 → wasteful, If it's timed well (~0.25–0.6), it can reach close to 1.0., # If too late (> 0.6), effectiveness drops back down to 0.9
            recovered = max(self.machine_health, effectiveness) #  The machine only recovers if the maintenance is better than current health.
            reward += 0.4 * (recovered - self.machine_health)  # Reward is based on how much health is gained.(up to 0.4 per full recovery)

            self.machine_health = recovered
            self.product_quality = min(0.95, self.product_quality + 0.01)
            self.time_since_maintenance = 0.0
            reward -= 0.05  # small time/effort cost for maintenance

        # Health decay (mild and consistent, worsens if overdue)
        health_decay = 0.01   #machine health decays by 0.01 every step
        if self.time_since_maintenance > 0.6:
            health_decay *= 1.5  # higher decay if long overdue

        self.machine_health = max(0.0, self.machine_health - health_decay)

        # Quality degradation if health is poor
        if self.machine_health < 0.9:
            degradation = (0.9 - self.machine_health) * 0.5
            self.product_quality = max(0.7, self.product_quality - degradation)
        else:
            self.product_quality = min(0.95, self.product_quality + 0.01)

        # Optimal quality shaping, Ideal quality occurs when param1 is around 0.7 and param2 around 0.3
        # This is a simple model; in practice, you might use a more complex function
        optimal_quality = 0.8 + 0.2 * np.exp(-((self.param1 - 0.7)**2 / 0.01 + (self.param2 - 0.3)**2 / 0.01)) # This creates a peak around (0.7, 0.3), smooth decline away from it and more realsistic simulation of tolerances

        self.product_quality = np.clip(optimal_quality, 0.0, 1.0)

        # Reward shaping
        reward += (self.product_quality - 0.6) * 2
        reward -= 0.05 * self.time_since_maintenance

        # Penalties for failure conditions
        terminated = False
        if self.machine_health <= 0.0 or self.product_quality < 0.4:
            reward -= 1.0
            terminated = True

        truncated = self.steps >= self.max_steps # Truncate if max steps reached

        # Clip reward to stabilize DQN, as large rewards can destabilize training
        reward = np.clip(reward, -1.0, 1.0)

        # Logging
        self.metrics['healths'].append(self.machine_health)
        self.metrics['qualities'].append(self.product_quality)
        self.metrics['param1s'].append(self.param1)
        self.metrics['param2s'].append(self.param2)
        self.metrics['rewards'].append(reward)

        return self._get_obs(), reward, terminated, truncated, {
            "health": self.machine_health,
            "quality": self.product_quality,
            "maintenance_timer": self.time_since_maintenance
        }

    def render(self, mode='human'):
        print(
            f"Step {self.steps} | "
            f"Health: {self.machine_health:.2f}, "
            f"Quality: {self.product_quality:.2f}, "
            f"Time Since Maintenance: {self.time_since_maintenance:.2f}, "
            f"Param1: {self.param1:.2f}, Param2: {self.param2:.2f}"
        )

    def setup_live_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(2, 2, figsize=(10, 6))
        self.fig.suptitle("Industrial Automation Env Monitoring", fontsize=16)

    def update_live_plot(self):
        m = self.metrics

        self.ax[0, 0].cla()
        self.ax[0, 0].plot(m['healths'], color='red')
        self.ax[0, 0].set_title("Machine Health")
        self.ax[0, 0].set_ylim(0, 1)

        self.ax[0, 1].cla()
        self.ax[0, 1].plot(m['qualities'], color='green')
        self.ax[0, 1].set_title("Product Quality")
        self.ax[0, 1].set_ylim(0, 1)

        self.ax[1, 0].cla()
        self.ax[1, 0].plot(m['param1s'], color='blue', label='Param1')
        self.ax[1, 0].plot(m['param2s'], color='orange', label='Param2')
        self.ax[1, 0].set_title("Control Parameters")
        self.ax[1, 0].set_ylim(0, 1)
        self.ax[1, 0].legend()

        self.ax[1, 1].cla()
        self.ax[1, 1].plot(m['rewards'], color='purple')
        self.ax[1, 1].set_title("Reward Over Time")
        self.ax[1, 1].set_ylim(-2, 2)

        plt.tight_layout()
        plt.pause(0.1)

    def finalize_plot(self):
        plt.ioff()
        plt.show()
