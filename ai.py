import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy
import time  # Added for time tracking
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
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Calculate the flattened size
        flat_size = 64 * board_size * board_size
        
        self.fc_layers = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc_layers(x)


class ReplayBuffer:
    """Memory buffer for experience replay."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class AbaloneAI:
    """AI agent for playing Abalone using Deep Q-Learning."""
    
    def __init__(self, player: Player, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 gamma=0.99, learning_rate=0.001, batch_size=64):
        self.player = player
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma  # Discount factor
        self.batch_size = batch_size
        
        # Initialize the models and move them to the appropriate device
        self.model = DQNModel().to(device)
        self.target_model = copy.deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.memory = ReplayBuffer()
        
        # Action mapping
        self.action_map = []  # Will be populated during training
        self.max_actions = 1000  # Maximum number of possible actions
    
    def update_action_map(self, valid_moves):
        """Update the mapping of action indices to actual moves."""
        self.action_map = valid_moves
        return {i: move for i, move in enumerate(valid_moves)}
    
    def select_action(self, game: AbaloneGame, training=True):
        """Select an action using epsilon-greedy policy."""
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
        
        # Get the actual move
        action = action_map[action_idx]
        
        # Decay epsilon
        if training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return action, action_idx
    
    def train(self, batch):
        """Train the model on a batch of experiences."""
        if len(batch) == 0:
            return
        
        # Unpack the batch
        states, action_indices, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(device)
        action_indices = torch.LongTensor(action_indices).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Compute current Q values
        current_q_values = self.model(states).gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values using target model
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_model(self):
        """Update the target model with the current model weights."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save_model(self, path):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path):
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


def train_ai(num_episodes=1000, target_update=10, save_interval=100, time_limit_minutes=5):
    """Train the AI through self-play."""
    game = AbaloneGame()
    
    # Create two AI agents
    black_ai = AbaloneAI(Player.BLACK)
    white_ai = AbaloneAI(Player.WHITE)
    
    # Training stats
    stats = {
        'episode_rewards': [],
        'black_wins': 0,
        'white_wins': 0,
        'draws': 0
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
        episode_reward = 0
        
        # Keep track of the game trajectory
        trajectory = []
        
        while not game.game_over:
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
            
            # Calculate reward
            # Simple reward scheme: win = 1, loss = -1, intermediate = small reward based on marbles pushed
            reward = 0
            if game.game_over:
                if game.winner == current_player_ai.player:
                    reward = 1  # Win
                else:
                    reward = -1  # Loss
            else:
                # Small reward for pushing off opponent marbles
                opponent = current_player_ai.player.opponent()
                reward = 0.1 * game.pushed_off[opponent]
            
            episode_reward += reward
            
            # Get the next state
            next_state = game.get_state_representation()
            
            # Store experience in memory
            current_player_ai.memory.add(
                current_state, action_idx, reward, next_state, game.game_over
            )
            
            # Store in trajectory for opponent learning
            trajectory.append((
                current_state, action_idx, reward, next_state, game.game_over, current_player_ai.player
            ))
            
            # Train the model if enough samples are available
            if len(current_player_ai.memory) >= current_player_ai.batch_size:
                batch = current_player_ai.memory.sample(current_player_ai.batch_size)
                current_player_ai.train(batch)
        
        # Use trajectory data for opponent learning
        for state, action_idx, reward, next_state, done, player in trajectory:
            # Add experience to opponent's memory with inverted reward
            # (what's good for one player is bad for the other)
            opponent_ai = white_ai if player == Player.BLACK else black_ai
            opponent_ai.memory.add(state, action_idx, -reward, next_state, done)
            
            # Train opponent's model if enough samples
            if len(opponent_ai.memory) >= opponent_ai.batch_size:
                batch = opponent_ai.memory.sample(opponent_ai.batch_size)
                opponent_ai.train(batch)
        
        # Update stats
        stats['episode_rewards'].append(episode_reward)
        if game.winner == Player.BLACK:
            stats['black_wins'] += 1
        elif game.winner == Player.WHITE:
            stats['white_wins'] += 1
        else:
            stats['draws'] += 1
        
        # Update target networks periodically
        if episode % target_update == 0:
            black_ai.update_target_model()
            white_ai.update_target_model()
        
        # Save models periodically
        if episode % save_interval == 0:
            black_ai.save_model(f"black_ai_episode_{episode}.pth")
            white_ai.save_model(f"white_ai_episode_{episode}.pth")
        
        # Print progress more frequently to show activity
        if episode % 5 == 0:
            elapsed_minutes = (time.time() - start_time) / 60
            print(f"Episode {episode}/{num_episodes} ({elapsed_minutes:.1f} min elapsed), "
                  f"Black wins: {stats['black_wins']}, "
                  f"White wins: {stats['white_wins']}, "
                  f"Draws: {stats['draws']}")
    
    # Print total training time
    total_time = (time.time() - start_time) / 60
    print(f"Total training time: {total_time:.2f} minutes")
    
    return black_ai, white_ai, stats


def evaluate_ai(black_ai, white_ai, num_games=100):
    """Evaluate trained AI models by playing games without exploration."""
    game = AbaloneGame()
    
    stats = {
        'black_wins': 0,
        'white_wins': 0,
        'draws': 0,
        'avg_game_length': 0
    }
    
    total_turns = 0
    
    for i in range(num_games):
        game.reset()
        
        while not game.game_over:
            current_player_ai = black_ai if game.current_player == Player.BLACK else white_ai
            
            # Select an action without exploration
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
        elif game.winner == Player.WHITE:
            stats['white_wins'] += 1
        else:
            stats['draws'] += 1
        
        total_turns += game.turn_count
    
    stats['avg_game_length'] = total_turns / num_games
    
    print(f"Evaluation results ({num_games} games):")
    print(f"Black wins: {stats['black_wins']} ({stats['black_wins']/num_games*100:.1f}%)")
    print(f"White wins: {stats['white_wins']} ({stats['white_wins']/num_games*100:.1f}%)")
    print(f"Draws: {stats['draws']} ({stats['draws']/num_games*100:.1f}%)")
    print(f"Average game length: {stats['avg_game_length']:.1f} turns")
    
    return stats


def main():
    # Train AI models with time limit
    print("Training AI models (will stop after 5 minutes for testing)...")
    black_ai, white_ai, training_stats = train_ai(num_episodes=1000, time_limit_minutes=5)
    
    # Save models for testing
    black_ai.save_model("black_ai_test.pth")
    white_ai.save_model("white_ai_test.pth")
    
    # Evaluate AI models with a small number of games for quick testing
    print("Quick evaluation for testing...")
    evaluation_stats = evaluate_ai(black_ai, white_ai, num_games=10)
    print("Training and quick evaluation completed. Models saved for further testing.")
    
    continue_training = input("Continue training for full 1000 episodes? (y/n): ")
    if continue_training.lower() == 'y':
        print("Resuming training for full duration...")
        black_ai, white_ai, training_stats = train_ai(num_episodes=1000, time_limit_minutes=0)  # 0 means no time limit
        
        # Save final models
        black_ai.save_model("black_ai_final.pth")
        white_ai.save_model("white_ai_final.pth")
        
        # Full evaluation
        print("Performing full evaluation...")
        evaluation_stats = evaluate_ai(black_ai, white_ai, num_games=100)
    
    print("Process completed.")

if __name__ == "__main__":
    main()
# The code above is a complete implementation of a DQN-based AI for the Abalone game.