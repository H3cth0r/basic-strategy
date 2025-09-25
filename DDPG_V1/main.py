import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import time # Import time for adding a small delay during rendering

# --- Device Configuration ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1    = nn.Linear(state_dim, 400)
        self.layer_2    = nn.Linear(400, 300)
        self.layer_3    = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x   = F.relu(self.layer_1(x))
        x   = F.relu(self.layer_2(x))
        x   = self.max_action * torch.tanh(self.layer_3(x))
        return x
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1    = nn.Linear(state_dim + action_dim, 400)
        self.layer_2    = nn.Linear(400, 300)
        self.layer_3    = nn.Linear(300, 1)

    def forward(self, x, u):
        x   = F.relu(self.layer_1(torch.cat([x, u], 1)))
        x   = F.relu(self.layer_2(x))
        x   = self.layer_3(x)
        return x


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage    = []
        self.max_size   = max_size
        self.ptr        = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.asarray(X))
            y.append(np.asarray(Y))
            u.append(np.asarray(U))
            r.append(np.asarray(R))
            d.append(np.asarray(D))
            
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu             = mu
        self.theta          = theta
        self.sigma          = max_sigma
        self.max_sigma      = max_sigma
        self.min_sigma      = min_sigma
        self.decay_period   = decay_period
        self.action_dim     = action_space.shape[0]
        self.low            = action_space.low
        self.high           = action_space.high
        self.reset()

    def reset(self):
        self.state  = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x           = self.state
        dx          = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state  = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, device):
        self.device = device
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer()
        self.noise = OUNoise(env.action_space)

    def select_action(self, state):
       state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
       return self.actor(state).cpu().data.numpy().flatten()

    def update(self, batch_size, gamma=0.99, tau=0.005):
        x, y, u, r, d = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(x).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        done = torch.FloatTensor(1-d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)

        # --- Critic Loss ---
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (done * gamma * target_Q).detach()
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Loss ---
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft Target Updates ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


# ===================================================================================
# +++ NEW: Evaluation Function +++
# ===================================================================================
def evaluate_policy(agent, env_name, eval_episodes=1):
    """Runs policy for X episodes and returns average reward, rendering the environment."""
    # Important: Create a new environment for evaluation, with rendering enabled
    eval_env = gym.make(env_name, render_mode="human")
    
    total_reward = 0.
    for i in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Select action deterministically (no noise)
            action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            # Optional: add a small delay to make rendering smoother
            time.sleep(0.005)
        
        total_reward += episode_reward
        print(f"Evaluation Episode {i+1}: Reward = {episode_reward:.2f}")

    eval_env.close()
    avg_reward = total_reward / eval_episodes
    return avg_reward
# ===================================================================================


if __name__ == "__main__":
    env_name = "Pendulum-v1"
    # The training environment does not need rendering
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = DDPG(state_dim, action_dim, max_action, device)
    
    # --- CHANGE: Added a frequency for evaluation ---
    EVAL_FREQ = 50  # Evaluate and render every 50 episodes
    
    for i in range(2000):
        state, _ = env.reset()
        agent.noise.reset()
        episode_reward = 0
        for t in range(500):
            action = agent.select_action(state)
            action = agent.noise.get_action(action, t)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add((state, next_state, action, reward, float(done)))
            
            if len(agent.replay_buffer.storage) > 2000:
                agent.update(100)
            
            state = next_state
            episode_reward += reward

            if done:
                break
        print(f"Episode: {i+1}, Total Reward: {episode_reward:.2f}")

        # --- CHANGE: Perform evaluation and rendering periodically ---
        if (i + 1) % EVAL_FREQ == 0:
            print("\n---------------------------------------")
            print(f"Running evaluation for Episode {i+1}...")
            avg_reward = evaluate_policy(agent, env_name)
            print(f"Evaluation Avg. Reward: {avg_reward:.2f}")
            print("---------------------------------------\n")
            
    env.close()
