import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy
import time
from typing import List, Tuple

# Import the Abalone game implementation
from abalone_game import AbaloneGame, Player

# Set up device for M1 GPU (MPS) or fall back to CPU
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS (M1 GPU) device: {device}")
else:
    print("MPS (M1 GPU) not available, using CPU")

class DQNModel(nn.Module):
    """
    Deep Q-Network for Abalone.
    Input: Board state representation
    Output: Q-values for each possible move
    """
    def __init__(self, input_channels=3, board_size=9, num_actions=1000):
        super(DQNModel, self).__init__()
        
        # Enhanced network architecture with residual connections
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Residual block
        self.res_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res_bn1 = nn.BatchNorm2d(128)
        self.res_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res_bn2 = nn.BatchNorm2d(128)
        
        # Value and advantage streams (Dueling DQN architecture)
        flat_size = 128 * board_size * board_size
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Convolutional layers with batch normalization
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Residual block
        residual = x
        x = self.relu(self.res_bn1(self.res_conv1(x)))
        x = self.res_bn2(self.res_conv2(x))
        x = self.relu(x + residual)  # Skip connection
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dueling DQN: split into value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for more efficient learning."""
    
    def __init__(self, capacity=50000, alpha=0.6, beta=0.4, beta_increment=0.0001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta = beta    # Importance sampling correction (0 = no correction, 1 = full)
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.next_idx = 0
    
    def add(self, state, action_idx, reward, next_state, done):
        """Add experience to buffer with maximum priority to ensure it's sampled."""
        experience = (state, action_idx, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.next_idx] = experience
        
        # New experiences get max priority to ensure they're sampled
        self.priorities[self.next_idx] = self.max_priority
        self.next_idx = (self.next_idx + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of experiences based on their priorities."""
        if len(self.buffer) == 0:
            return [], [], []
        
        # Increase beta for more accurate bias correction over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities from priorities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probabilities)
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, action_indices, rewards, next_states, dones = zip(*samples)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        return (
            np.array(states), 
            np.array(action_indices), 
            np.array(rewards), 
            np.array(next_states), 
            np.array(dones),
            indices,
            weights
        )
    
    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # Small constant to prevent zero priority
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self):
        return len(self.buffer)


class AbaloneAI:
    """AI agent for playing Abalone using enhanced Deep Q-Learning."""
    
    def __init__(self, player: Player, epsilon_start=1.0, epsilon_end=0.05, 
                 epsilon_decay_steps=50000, gamma=0.99, learning_rate=0.0001, 
                 batch_size=128, n_step=3):
        self.player = player
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.steps = 0
        self.gamma = gamma  # Discount factor
        self.batch_size = batch_size
        self.n_step = n_step  # For n-step returns
        
        # Initialize the models and move them to the appropriate device
        self.model = DQNModel().to(device)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Huber loss for better handling of outliers compared to MSE
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # For prioritized replay
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)
        
        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer()
        
        # Action mapping
        self.action_map = []
        self.max_actions = 1000
        
        # N-step return buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Training stats
        self.training_stats = {
            'losses': [],
            'rewards': [],
            'q_values': []
        }
    
    def update_action_map(self, valid_moves):
        """Update the mapping of action indices to actual moves."""
        self.action_map = valid_moves
        return {i: move for i, move in enumerate(valid_moves)}
    
    def select_action(self, game: AbaloneGame, training=True):
        """Select an action using epsilon-greedy policy with decaying epsilon."""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None, None  # Return tuple for consistency
        
        # Update action mapping
        action_map = self.update_action_map(valid_moves)
        
        # Convert state to tensor and move to device
        state = game.get_state_representation()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dimension
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Exploration: select a random action
            action_idx = random.randrange(len(valid_moves))
        else:
            # Exploitation: select the best action according to the model
            with torch.no_grad():
                q_values = self.model(state_tensor)
                
                # Filter to only valid actions
                valid_q_values = []
                valid_indices = []
                
                for i in range(len(valid_moves)):
                    if i < self.max_actions:
                        valid_q_values.append(q_values[0, i].item())
                        valid_indices.append(i)
                
                # Select the action with the highest Q-value
                max_idx = valid_indices[np.argmax(valid_q_values)]
                action_idx = max_idx
                
                # Track average Q-values for monitoring learning progress
                if training:
                    self.training_stats['q_values'].append(np.max(valid_q_values))
        
        # Get the actual move
        action = action_map[action_idx]
        
        # Decay epsilon using linear annealing
        if training:
            self.steps += 1
            self.epsilon = max(
                self.epsilon_end, 
                self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.steps / self.epsilon_decay_steps)
            )
        
        return action, action_idx
    
    def calculate_reward(self, game: AbaloneGame):
        """Enhanced reward function with strategic considerations."""
        if game.game_over:
            if game.winner == self.player:
                return 10.0  # Win
            elif game.winner is None:
                return 0.0   # Draw
            else:
                return -10.0  # Loss
        
        # Calculate board control reward (center control is valuable in Abalone)
        board = game.board
        center_control = 0
        center_positions = [
            (3, 3), (3, 4), (3, 5), 
            (4, 3), (4, 4), (4, 5), 
            (5, 3), (5, 4), (5, 5)
        ]
        
        for pos in center_positions:
            r, c = pos  # Unpack the tuple to access row and column separately
            if board[r, c] == self.player.value:
                center_control += 0.05
        
        # Group cohesion reward - connected marbles are stronger
        cohesion_reward = 0
        player_marbles = [
            (r, c) for r in range(9) for c in range(9) 
            if board[r, c] == self.player.value
        ]
        
        # Count adjacent friendly marbles
        for r, c in player_marbles:
            neighbors = game.get_adjacent_positions((r, c))
            for nr, nc in neighbors:
                if 0 <= nr < 9 and 0 <= nc < 9 and board[nr, nc] == self.player.value:
                    cohesion_reward += 0.01
        
        # Marble advantage reward
        player_count = np.sum(board == self.player.value)
        opponent_count = np.sum(board == self.player.opponent().value)
        material_advantage = 0.1 * (player_count - opponent_count)
        
        # Reward for pushing off opponent marbles
        pushed_off_reward = 0.5 * game.pushed_off[self.player.opponent()]
        
        # Total reward
        total_reward = center_control + cohesion_reward + material_advantage + pushed_off_reward
        
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
        
        # Compute current Q values
        current_q_values = self.model(states).gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use current network to select actions and target network to evaluate them
        with torch.no_grad():
            # Select actions using the current network
            next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            # Evaluate Q-values using the target network
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
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


def train_ai(num_episodes=1000, target_update_freq=10, save_interval=100, time_limit_minutes=120):
    """Train the AI through self-play with enhanced techniques."""
    game = AbaloneGame()
    
    # Create two AI agents
    black_ai = AbaloneAI(Player.BLACK)
    white_ai = AbaloneAI(Player.WHITE)
    
    # Training stats
    stats = {
        'episode_rewards': [],
        'black_wins': 0,
        'white_wins': 0,
        'draws': 0,
        'avg_loss': [],
        'avg_q_value': []
    }
    
    start_time = time.time()
    time_limit_seconds = time_limit_minutes * 60
    
    for episode in range(num_episodes):
        # Check if time limit has been reached
        if time.time() - start_time > time_limit_seconds:
            print(f"\nTime limit of {time_limit_minutes} minutes reached. Stopping training.")
            print(f"Completed {episode} episodes.")
            break
            
        game.reset()
        episode_black_reward = 0
        episode_white_reward = 0
        
        # Track episode metrics
        episode_losses = []
        moves_count = 0
        
        # Clear n-step buffers at the start of each episode
        black_ai.n_step_buffer.clear()
        white_ai.n_step_buffer.clear()
        
        while not game.game_over:
            moves_count += 1
            
            # Handle very long games to prevent infinite loops
            if moves_count > 200:
                game.game_over = True
                game.winner = None
                break
            
            current_player_ai = black_ai if game.current_player == Player.BLACK else white_ai
            opponent_ai = white_ai if game.current_player == Player.BLACK else black_ai
            
            # Get the current state
            current_state = game.get_state_representation()
            
            # Select an action
            action, action_idx = current_player_ai.select_action(game)
            
            if action is None:
                # No valid moves, game is a draw
                game.game_over = True
                game.winner = None
                break
            
            # Make the move
            line, direction = action
            game.make_move(line, direction)
            
            # Calculate enhanced reward
            reward = current_player_ai.calculate_reward(game)
            
            # Track total rewards
            if current_player_ai.player == Player.BLACK:
                episode_black_reward += reward
            else:
                episode_white_reward += reward
            
            # Get the next state
            next_state = game.get_state_representation()
            
            # Add experience to n-step buffer
            current_player_ai.add_to_n_step_buffer(
                current_state, action_idx, reward, next_state, game.game_over
            )
            
            # Train the model
            if len(current_player_ai.memory) >= current_player_ai.batch_size:
                loss = current_player_ai.train()
                episode_losses.append(loss)
            
            # Soft update target model more frequently for stability
            current_player_ai.soft_update_target_model(tau=0.001)
            
            # Train the opponent with negative reward
            if len(opponent_ai.memory) >= opponent_ai.batch_size:
                opponent_ai.train()
        
        # Add final experiences from n-step buffer
        for i in range(len(black_ai.n_step_buffer)):
            state, action_idx, reward, next_state, done = black_ai.n_step_buffer[i]
            sum_reward = reward
            # Calculate future rewards
            for j in range(i + 1, len(black_ai.n_step_buffer)):
                sum_reward += black_ai.gamma ** (j - i) * black_ai.n_step_buffer[j][2]
            black_ai.memory.add(state, action_idx, sum_reward, next_state, done)
        
        for i in range(len(white_ai.n_step_buffer)):
            state, action_idx, reward, next_state, done = white_ai.n_step_buffer[i]
            sum_reward = reward
            # Calculate future rewards
            for j in range(i + 1, len(white_ai.n_step_buffer)):
                sum_reward += white_ai.gamma ** (j - i) * white_ai.n_step_buffer[j][2]
            white_ai.memory.add(state, action_idx, sum_reward, next_state, done)
        
        # Update learning rate scheduler
        black_ai.scheduler.step()
        white_ai.scheduler.step()
        
        # Update stats
        stats['episode_rewards'].append((episode_black_reward, episode_white_reward))
        if game.winner == Player.BLACK:
            stats['black_wins'] += 1
        elif game.winner == Player.WHITE:
            stats['white_wins'] += 1
        else:
            stats['draws'] += 1
        
        # Track average loss and Q-values
        if episode_losses:
            stats['avg_loss'].append(np.mean(episode_losses))
        if black_ai.training_stats['q_values']:
            recent_q_values = black_ai.training_stats['q_values'][-moves_count:]
            if recent_q_values:
                stats['avg_q_value'].append(np.mean(recent_q_values))
        
        # Print progress with detailed metrics
        if episode % 10 == 0:
            elapsed_minutes = (time.time() - start_time) / 60
            avg_loss = np.mean(stats['avg_loss'][-10:]) if stats['avg_loss'] else 0
            avg_q = np.mean(stats['avg_q_value'][-10:]) if stats['avg_q_value'] else 0
            
            print(f"Episode {episode}/{num_episodes} ({elapsed_minutes:.1f} min) | " 
                  f"Black/White/Draw: {stats['black_wins']}/{stats['white_wins']}/{stats['draws']} | "
                  f"Epsilon: {black_ai.epsilon:.3f} | Avg Loss: {avg_loss:.4f} | Avg Q: {avg_q:.4f}")
        
        # Save models periodically
        if episode % save_interval == 0:
            black_ai.save_model(f"black_ai_episode_{episode}.pth")
            white_ai.save_model(f"white_ai_episode_{episode}.pth")
    
    # Save final models
    black_ai.save_model("black_ai_final.pth")
    white_ai.save_model("white_ai_final.pth")
    
    # Print total training time
    total_time = (time.time() - start_time) / 60
    print(f"Total training time: {total_time:.2f} minutes")
    
    return black_ai, white_ai, stats


def evaluate_ai(black_ai, white_ai, num_games=100):
    """Evaluate trained AI models with detailed metrics."""
    game = AbaloneGame()
    
    stats = {
        'black_wins': 0,
        'white_wins': 0,
        'draws': 0,
        'avg_game_length': 0,
        'pushed_marbles': {'black': 0, 'white': 0},
        'center_control': {'black': [], 'white': []},
        'win_margin': []
    }
    
    total_turns = 0
    start_time = time.time()
    
    for i in range(num_games):
        game.reset()
        game_center_control = {'black': 0, 'white': 0}
        
        while not game.game_over:
            # Calculate center control
            center_positions = [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5)]
            black_center = sum(1 for pos in center_positions if game.board[pos] == Player.BLACK.value)
            white_center = sum(1 for pos in center_positions if game.board[pos] == Player.WHITE.value)
            game_center_control['black'] += black_center
            game_center_control['white'] += white_center
            
            current_player_ai = black_ai if game.current_player == Player.BLACK else white_ai
            
            # Select action without exploration
            action, _ = current_player_ai.select_action(game, training=False)
            
            if action is None:
                # No valid moves, game is a draw
                game.game_over = True
                game.winner = None
                break
            
            # Make the move
            line, direction = action
            game.make_move(line, direction)
            
            # Limit game length to prevent infinite games
            if game.turn_count > 200:
                game.game_over = True
                game.winner = None
                break
        
        # Update stats
        if game.winner == Player.BLACK:
            stats['black_wins'] += 1
            # Calculate win margin (difference in marbles)
            margin = game.pushed_off[Player.WHITE] - game.pushed_off[Player.BLACK]
            stats['win_margin'].append(margin)
        elif game.winner == Player.WHITE:
            stats['white_wins'] += 1
            # Calculate win margin (difference in marbles)
            margin = game.pushed_off[Player.BLACK] - game.pushed_off[Player.WHITE]
            stats['win_margin'].append(margin)
        else:
            stats['draws'] += 1
        
        # Update pushed marbles stats
        stats['pushed_marbles']['black'] += game.pushed_off[Player.BLACK]
        stats['pushed_marbles']['white'] += game.pushed_off[Player.WHITE]
        
        # Update center control stats
        stats['center_control']['black'].append(game_center_control['black'] / game.turn_count if game.turn_count > 0 else 0)
        stats['center_control']['white'].append(game_center_control['white'] / game.turn_count if game.turn_count > 0 else 0)
        
        total_turns += game.turn_count
        
        # Print progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Evaluated {i + 1}/{num_games} games ({elapsed:.1f} sec)")
    
    # Calculate average game length
    stats['avg_game_length'] = total_turns / num_games
    
    # Average center control
    stats['center_control']['black'] = np.mean(stats['center_control']['black'])
    stats['center_control']['white'] = np.mean(stats['center_control']['white'])
    
    # Average win margin
    stats['avg_win_margin'] = np.mean(stats['win_margin']) if stats['win_margin'] else 0
    
    # Print detailed evaluation results
    print("\nDetailed Evaluation Results:")
    print(f"Games played: {num_games}")
    print(f"Black wins: {stats['black_wins']} ({stats['black_wins']/num_games*100:.1f}%)")
    print(f"White wins: {stats['white_wins']} ({stats['white_wins']/num_games*100:.1f}%)")
    print(f"Draws: {stats['draws']} ({stats['draws']/num_games*100:.1f}%)")
    print(f"Average game length: {stats['avg_game_length']:.1f} turns")
    print(f"Average marbles pushed off - Black: {stats['pushed_marbles']['black']/num_games:.2f}, "
          f"White: {stats['pushed_marbles']['white']/num_games:.2f}")
    print(f"Average center control - Black: {stats['center_control']['black']:.2f}, "
          f"White: {stats['center_control']['white']:.2f}")
    if stats['win_margin']:
        print(f"Average win margin: {stats['avg_win_margin']:.2f} marbles")
    
    return stats


def main():
    # Allow loading existing models to continue training
    load_existing = input("Load existing models for continued training? (y/n): ")
    
    if load_existing.lower() == 'y':
        black_model_path = input("Enter path to black AI model (leave empty for default 'black_ai_final.pth'): ")
        white_model_path = input("Enter path to white AI model (leave empty for default 'white_ai_final.pth'): ")
        
        black_model_path = black_model_path or "black_ai_final.pth"
        white_model_path = white_model_path or "white_ai_final.pth"
        
        try:
            black_ai = AbaloneAI(Player.BLACK)
            white_ai = AbaloneAI(Player.WHITE)
            black_ai.load_model(black_model_path)
            white_ai.load_model(white_model_path)
            print(f"Models loaded successfully. Black AI at step {black_ai.steps}, White AI at step {white_ai.steps}")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Creating new models instead.")
            black_ai = AbaloneAI(Player.BLACK)
            white_ai = AbaloneAI(Player.WHITE)
    else:
        print("Creating new models...")
        black_ai = AbaloneAI(Player.BLACK)
        white_ai = AbaloneAI(Player.WHITE)
    
    # Ask about training
    do_training = input("Start training process? (y/n): ")
    if do_training.lower() == 'y':
        num_episodes = int(input("Enter number of training episodes (default 1000): ") or "1000")
        time_limit = int(input("Enter training time limit in minutes (default 120): ") or "120")
        save_interval = int(input("Enter save interval in episodes (default 100): ") or "100")
        
        print(f"Starting training for {num_episodes} episodes (max {time_limit} minutes)...")
        black_ai, white_ai, train_stats = train_ai(
            num_episodes=num_episodes, 
            save_interval=save_interval,
            time_limit_minutes=time_limit
        )
        
        print("\nTraining completed!")
        print(f"Final results - Black wins: {train_stats['black_wins']}, "
              f"White wins: {train_stats['white_wins']}, "
              f"Draws: {train_stats['draws']}")
    
    # Ask about evaluation
    do_eval = input("Evaluate models? (y/n): ")
    if do_eval.lower() == 'y':
        num_eval_games = int(input("Enter number of evaluation games (default 100): ") or "100")
        print(f"Evaluating models over {num_eval_games} games...")
        eval_stats = evaluate_ai(black_ai, white_ai, num_games=num_eval_games)
    
    # Ask about saving models if they've been trained
    if do_training.lower() == 'y':
        save_models = input("Save trained models? (y/n): ")
        if save_models.lower() == 'y':
            black_save_path = input("Enter path for saving black AI model (leave empty for default 'black_ai_final.pth'): ")
            white_save_path = input("Enter path for saving white AI model (leave empty for default 'white_ai_final.pth'): ")
            
            black_save_path = black_save_path or "black_ai_final.pth"
            white_save_path = white_save_path or "white_ai_final.pth"
            
            black_ai.save_model(black_save_path)
            white_ai.save_model(white_save_path)
            print(f"Models saved to {black_save_path} and {white_save_path}")

if __name__ == "__main__":
    main()