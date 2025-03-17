# abalone_ai.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy
from typing import List, Tuple

from abalone_game import AbaloneGame, Player  # Import the Abalone game implementation
from dqn_model import DQNModel  # Import the DQN model
from replay_buffer import PrioritizedReplayBuffer  # Import the replay buffer

# Set up device: CUDA if available, else MPS, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using CUDA device: {device}")
elif device.type == "mps":
    print(f"Using MPS (M1 GPU) device: {device}")
else:
    print("Using CPU")


class AbaloneAI:
    """AI agent for playing Abalone using enhanced Deep Q-Learning."""

    def __init__(self, player: Player, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_steps=50000, gamma=0.99, learning_rate=0.0001,
                 batch_size=128, n_step=3, num_episodes=1000): # Added num_episodes for Cosine Annealing
        self.player = player
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.steps = 0
        self.gamma = gamma  # Discount factor
        self.batch_size = batch_size
        self.n_step = n_step  # For n-step returns
        self.num_episodes = num_episodes # For Cosine Annealing

        # Initialize the models and move them to the appropriate device
        self.model = DQNModel().to(device)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Huber loss for better handling of outliers compared to MSE
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # For prioritized replay

        # Cosine Annealing Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_episodes, eta_min=0) # Use num_episodes as T_max

        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer()

        # Training stats
        self.training_stats = {
            'losses': [],
            'rewards': [],
            'q_values': []
        }
         # N-step return buffer
        self.n_step_buffer = deque(maxlen=n_step)

    def select_action(self, game: AbaloneGame, training=True):
        """Select an action using epsilon-greedy policy with decaying epsilon."""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None, None  # Return tuple for consistency

        # Convert state to tensor and move to device
        state = game.get_state_representation()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dimension
        num_actions = len(valid_moves)  # Dynamically determine the number of actions

        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Exploration: select a random action
            action_idx = random.randrange(num_actions)
        else:
            # Exploitation: select the best action according to the model
            with torch.no_grad():
                q_values = self.model(state_tensor, num_actions)
                action_idx = torch.argmax(q_values).item()  # Select the action with the highest Q-value
                self.training_stats['q_values'].append(q_values.max().item())  # Track Q Value

        # Get the actual move
        action = valid_moves[action_idx]

        # Decay epsilon using exponential decay
        if training:
            self.steps += 1
            decay_rate = (self.epsilon_end / self.epsilon_start) ** (1 / self.epsilon_decay_steps)
            self.epsilon = max(self.epsilon_end, self.epsilon_start * (decay_rate ** self.steps))

        return action, action_idx

    def potential(self, game: AbaloneGame):
        """Potential function for reward shaping."""
        player_count = sum(1 for p in game.board.values() if p == self.player)
        opponent_count = sum(1 for p in game.board.values() if p == self.player.opponent())
        material_advantage = player_count - opponent_count

        center_positions = [
            (-1, 1, 0), (0, 1, -1), (1, 1, -2),
            (-1, 0, 1), (0, 0, 0), (1, 0, -1),
            (-1, -1, 2), (0, -1, 1), (1, -1, 0)
        ]
        center_control = sum(1 for pos in center_positions if game.is_valid_position(pos) and game.board.get(pos) == self.player)

        return material_advantage + 0.5 * center_control # Weighted sum of material and center control

    def calculate_reward(self, game: AbaloneGame, previous_game_state=None): # Added previous_game_state
        """Enhanced reward function with potential-based reward shaping."""
        current_potential = self.potential(game)
        previous_potential = self.potential(previous_game_state) if previous_game_state else current_potential # Use current potential if no previous state

        # Game outcome rewards with higher magnitudes
        if game.game_over:
            if game.winner == self.player:
                return 50.0  # Much higher reward for winning
            elif game.winner is None:
                return -1.0  # Small penalty for draws to discourage them
            else:
                return -30.0  # Significant penalty for losing

        # Get counts of marbles
        player_count = sum(1 for p in game.board.values() if p == self.player)
        opponent_count = sum(1 for p in game.board.values() if p == self.player.opponent())

        # Material advantage with higher weight
        material_advantage = 0.5 * (player_count - opponent_count)

        # Strong reward for pushing off opponent marbles
        pushed_off_reward = 2.0 * game.pushed_off[self.player.opponent()]

        # Penalty for losing own marbles
        own_losses_penalty = -2.5 * game.pushed_off[self.player]

        # Center control
        center_positions = [
            (-1, 1, 0), (0, 1, -1), (1, 1, -2),
            (-1, 0, 1), (0, 0, 0), (1, 0, -1),
            (-1, -1, 2), (0, -1, 1), (1, -1, 0)
        ]
        center_control = 0
        for pos in center_positions:
            if game.is_valid_position(pos) and game.board.get(pos) == self.player:
                center_control += 0.2  # Increased weight

        # Progress toward win condition (pushing 6 opponent marbles)
        progress_to_win = 0.3 * (game.pushed_off[self.player.opponent()] / 6.0)

        # Reward for making moves that push opponent marbles toward the edge
        edge_pressure = 0
        for pos, p in game.board.items():
            if p == self.player.opponent():
                # Calculate distance from center (0,0,0)
                distance = max(abs(pos[0]), abs(pos[1]), abs(pos[2]))
                edge_pressure += 0.05 * distance

        # Cohesion reward (groups of friendly marbles are stronger)
        cohesion_reward = 0
        player_marbles = [pos for pos, p in game.board.items() if p == self.player]
        for pos in player_marbles:
            friendly_neighbors = 0
            for direction in range(6):
                neighbor = game.get_neighbor(pos, direction)
                if game.is_valid_position(neighbor) and game.board.get(neighbor) == self.player:
                    friendly_neighbors += 1
            cohesion_reward += 0.05 * friendly_neighbors

        # Potential-based reward shaping component
        reward_shaping = self.gamma * current_potential - previous_potential

        # Total reward
        total_reward = (
            material_advantage +
            pushed_off_reward +
            own_losses_penalty +
            center_control +
            progress_to_win +
            edge_pressure +
            cohesion_reward +
            reward_shaping # Add reward shaping
        )

        return total_reward

    def add_to_n_step_buffer(self, state, action_idx, reward, next_state, done):
        """Add experience to n-step buffer."""
        self.n_step_buffer.append((state, action_idx, reward, next_state, done))

        # If buffer is not full yet, don't process
        if len(self.n_step_buffer) < self.n_step:
            return

        # Get the initial experience
        init_state, init_action, _, _, _ = self.n_step_buffer[0]

        # Compute n-step return
        n_step_reward = 0
        for i in range(self.n_step):
            n_step_reward += (self.gamma ** i) * self.n_step_buffer[i][2]

        # Get the final state
        _, _, _, final_next_state, final_done = self.n_step_buffer[-1]

        # Add n-step experience to memory
        self.memory.add(init_state, init_action, n_step_reward, final_next_state, final_done)

    def train(self, batch=None):
        """Train the model using Double DQN with prioritized experience replay."""
        if batch is None:
            if len(self.memory) < self.batch_size:
                return 0
            batch = self.memory.sample(self.batch_size)

        states, action_indices, rewards, next_states, dones, indices, weights = batch

        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(device)
        action_indices = torch.LongTensor(action_indices).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)

        # Get the number of actions based on states
        game = AbaloneGame()  #create new game to get valid moves
        valid_moves = game.get_valid_moves() # Get the number of moves
        num_actions = len(valid_moves) # Get number of possible actions

        # Compute current Q values
        current_q_values = self.model(states, num_actions).gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # Double DQN: use current network to select actions and target network to evaluate them
        with torch.no_grad():
            # Select actions using the current network
            next_q_values_for_actions = self.model(next_states, num_actions)
            next_actions = torch.argmax(next_q_values_for_actions, dim=1, keepdim=True)

            # Evaluate Q-values using the target network
            next_q_values = self.target_model(next_states, num_actions).gather(1, next_actions).squeeze(1)

            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values

        # Calculate TD errors for prioritized replay
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()

        # Compute weighted loss for prioritized replay
        losses = self.loss_fn(current_q_values, target_q_values)
        loss = (losses * weights).mean()

        # Update model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)

        # Track loss for monitoring
        loss_value = loss.item()
        self.training_stats['losses'].append(loss_value)

        return loss_value

    def soft_update_target_model(self, tau=0.001):
        """Soft update target model parameters: θ′ ← τθ + (1 − τ)θ′"""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_model(self, path):
        """Save the model to disk with training stats."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'training_stats': self.training_stats
        }, path)

    def load_model(self, path):
        """Load the model from disk with training stats."""
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.training_stats = checkpoint['training_stats']