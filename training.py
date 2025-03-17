# training.py

import time
import numpy as np

from abalone_game import AbaloneGame, Player
from abalone_ai import AbaloneAI


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