# abalone_ai.py (no changes needed in this file)
# dqn_model.py (no changes needed in this file)
# replay_buffer.py (no changes needed in this file)

# training.py
import time
import numpy as np
import multiprocessing as mp
import copy
import torch  # Import torch here

from abalone_game import AbaloneGame, Player
from abalone_ai import AbaloneAI

def self_play_worker(worker_id, num_episodes, ai_params, experience_queue):
    """Worker function for parallel self-play, using a queue to send experiences."""
    print(f"Worker {worker_id} starting...")
    # Set CUDA device for worker if available, else CPU - Important for multi-GPU setup if needed.
    worker_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Or "cuda:worker_id % num_gpus" for multi-GPU
    if worker_device.type == "cuda":
        print(f"Worker {worker_id} using CUDA device: {worker_device}")
    else:
        print(f"Worker {worker_id} using CPU")

    black_ai = AbaloneAI(Player.BLACK, device=worker_device, **ai_params) # Pass device to AI
    white_ai = AbaloneAI(Player.WHITE, device=worker_device, **ai_params) # Pass device to AI
    game = AbaloneGame()
    worker_stats = {
        'episode_rewards': [],
        'black_wins': 0,
        'white_wins': 0,
        'draws': 0,
        'losses': [], # Collect losses per worker
        'q_values': [] # Collect q_values per worker
    }

    for episode in range(num_episodes):
        game.reset()
        episode_black_reward = 0
        episode_white_reward = 0
        episode_losses = []
        moves_count = 0
        black_ai.n_step_buffer.clear()
        white_ai.n_step_buffer.clear()

        previous_game_state_black = None
        previous_game_state_white = None

        episode_experiences = [] # List to hold experiences for this episode

        while not game.game_over:
            moves_count += 1
            if moves_count > 200:
                game.game_over = True
                game.winner = None
                break

            current_player_ai = black_ai if game.current_player == Player.BLACK else white_ai
            opponent_ai = white_ai if game.current_player == Player.BLACK else black_ai

            current_state = game.get_state_representation()
            action, action_idx = current_player_ai.select_action(game)

            if action is None:
                game.game_over = True
                game.winner = None
                break

            line, direction = action
            current_game_state = copy.deepcopy(game)
            game.make_move(line, direction)
            reward = current_player_ai.calculate_reward(game, current_game_state)

            if current_player_ai.player == Player.BLACK:
                episode_black_reward += reward
                previous_game_state_black = current_game_state
                previous_game_state_white = None
            else:
                episode_white_reward += reward
                previous_game_state_white = current_game_state
                previous_game_state_black = None

            next_state = game.get_state_representation()

            # Store experience for this step in the episode
            experience = (current_state, action_idx, reward, next_state, game.game_over)
            episode_experiences.append((experience, current_player_ai)) # Store AI agent too

        # Process episode experiences for n-step returns and add to queue
        for i in range(len(episode_experiences)):
            experience, ai_agent = episode_experiences[i]
            state, action_idx, reward, next_state, done = experience
            ai_agent.add_to_n_step_buffer(state, action_idx, reward, next_state, done)

            if len(ai_agent.n_step_buffer) == ai_agent.n_step: # Only send when n-step buffer is full
                init_state, init_action, _, _, _ = ai_agent.n_step_buffer[0]
                n_step_reward = 0
                for j in range(ai_agent.n_step):
                    n_step_reward += (ai_agent.gamma ** j) * ai_agent.n_step_buffer[j][2]
                _, _, _, final_next_state, final_done = ai_agent.n_step_buffer[-1]
                n_step_experience = (init_state, init_action, n_step_reward, final_next_state, final_done)
                experience_queue.put(n_step_experience) # Put n-step experience into queue


        # After episode, train worker's AI (optional, can train only in main process)
        current_episode_losses = []
        for _ in range(len(black_ai.memory) // black_ai.batch_size): # Train multiple times per episode if enough data
            if len(black_ai.memory) >= black_ai.batch_size:
                loss = black_ai.train()
                current_episode_losses.append(loss)
            if len(white_ai.memory) >= white_ai.batch_size: # Train white too - for self-play learning
                white_ai.train() # No need to track loss for both for now

        if current_episode_losses:
            worker_stats['losses'].append(np.mean(current_episode_losses))
        if black_ai.training_stats['q_values']:
            worker_stats['q_values'].extend(black_ai.training_stats['q_values'][-moves_count:])


        worker_stats['episode_rewards'].append((episode_black_reward, episode_white_reward))
        if game.winner == Player.BLACK:
            worker_stats['black_wins'] += 1
        elif game.winner == Player.WHITE:
            worker_stats['white_wins'] += 1
        else:
            worker_stats['draws'] += 1

    print(f"Worker {worker_id} finished.")
    return worker_stats # Only return stats now


def experience_collector(experience_queue, main_black_ai, num_episodes):
    """Process to collect experiences from the queue and add to main AI's memory."""
    collected_experiences = 0
    print("Experience collector starting...")
    # Get main AI device from model's parameters instead of model itself
    main_device = next(main_black_ai.model.parameters()).device # Get device from model's parameters
    while collected_experiences < num_episodes * 200 * 4: # Max experiences roughly (episodes * max moves * workers). Adjust as needed.  *200*4 is a generous upper bound for moves * workers.
        try:
            experience = experience_queue.get(timeout=10) # Timeout to avoid infinite blocking if workers finish early
            # Move experience tensors to main device if they are not already there - crucial for CUDA
            state, action_idx, reward, next_state, done = experience
            state = state if isinstance(state, np.ndarray) else state.to(main_device) # No need to move numpy arrays
            next_state = next_state if isinstance(next_state, np.ndarray) else next_state.to(main_device) # No need to move numpy arrays
            main_black_ai.memory.add(state, action_idx, reward, next_state, done) # Add experience to main AI's memory
            collected_experiences += 1
        except mp.queues.Empty:
            print("Experience queue empty, collector exiting.")
            break
        except Exception as e: # Catch any potential errors in collector
            print(f"Collector process error: {e}")
            break # Exit collector process on error

    print(f"Experience collector finished, collected {collected_experiences} experiences.")


def train_ai_parallel(num_episodes=1000, target_update_freq=10, save_interval=100, time_limit_minutes=120, num_workers=4):
    """Train AI with parallel self-play using multiprocessing Queue for experience sharing."""
    start_time = time.time()
    time_limit_seconds = time_limit_minutes * 60

    # Set multiprocessing start method to 'spawn' - IMPORTANT for CUDA
    mp.set_start_method('spawn', force=True) # Use force=True to override if needed

    # Initialize main AI agents (will aggregate experiences and do the primary training)
    main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Main process device
    print(f"Main process using device: {main_device}")
    main_black_ai = AbaloneAI(Player.BLACK, device=main_device, num_episodes=num_episodes) # Main AI on main device
    main_white_ai = AbaloneAI(Player.WHITE, device=main_device, num_episodes=num_episodes) # Main AI on main device


    # AI parameters to be passed to workers (ensure they are serializable)
    ai_params = {
        'epsilon_start': main_black_ai.epsilon_start,
        'epsilon_end': main_black_ai.epsilon_end,
        'epsilon_decay_steps': main_black_ai.epsilon_decay_steps,
        'gamma': main_black_ai.gamma,
        'learning_rate': main_black_ai.optimizer.param_groups[0]['lr'], # Get current LR from main AI
        'batch_size': main_black_ai.batch_size,
        'n_step': main_black_ai.n_step,
        'num_episodes': num_episodes # Pass num_episodes for Cosine Annealing in workers too
    }

    stats = {
        'episode_rewards': [],
        'black_wins': 0,
        'white_wins': 0,
        'draws': 0,
        'avg_loss': [],
        'avg_q_value': []
    }
    all_worker_stats = [] # To collect stats from all workers

    experience_queue = mp.Queue() # Create the multiprocessing queue

    # Start experience collector process
    collector_process = mp.Process(target=experience_collector, args=(experience_queue, main_black_ai, num_episodes)) # Only collecting for black_ai for simplicity in this example, can extend to white_ai too.
    collector_process.start()

    for episode_group in range(num_episodes // num_workers): # Loop through episode groups
        if time.time() - start_time > time_limit_seconds:
            print(f"\nTime limit of {time_limit_minutes} minutes reached. Stopping training.")
            print(f"Completed {episode_group * num_workers} episodes.")
            break

        processes = []
        worker_episode_count = 1 # Episodes per worker in this group
        current_group_worker_stats = []

        for worker_id in range(num_workers):
            p = mp.Process(target=self_play_worker, args=(worker_id, worker_episode_count, ai_params, experience_queue)) # Pass queue to workers
            processes.append(p)
            p.start()

        for p in processes:
            p.join() # Wait for all workers to finish

        for worker_id in range(num_workers): # Collect stats from workers
            worker_stats = self_play_worker(worker_id, worker_episode_count, ai_params, experience_queue) # Re-run worker to get stats (stats are small, so ok to re-run for stats only) -  *Can improve this stat collection too if needed for very high performance*
            current_group_worker_stats.append(worker_stats)


        # Aggregate worker stats
        group_black_wins = sum(ws['black_wins'] for ws in current_group_worker_stats)
        group_white_wins = sum(ws['white_wins'] for ws in current_group_worker_stats)
        group_draws = sum(ws['draws'] for ws in current_group_worker_stats)
        group_episode_rewards = [reward for ws in current_group_worker_stats for reward in ws['episode_rewards']]
        group_losses = [loss for ws in current_group_worker_stats for loss in ws['losses'] if ws['losses']] # Filter empty loss lists
        group_q_values = [q_val for ws in current_group_worker_stats for q_val in ws['q_values'] if ws['q_values']] # Filter empty q_value lists
        all_worker_stats.extend(current_group_worker_stats) # Append worker stats for full stats at the end


        # Train main AI models (in main process, after collecting experiences via queue)
        episode_losses = []
        num_training_steps = len(main_black_ai.memory) // main_black_ai.batch_size # Train multiple times per group
        for _ in range(num_training_steps):
            if len(main_black_ai.memory) >= main_black_ai.batch_size:
                loss = main_black_ai.train()
                episode_losses.append(loss)
            if len(main_white_ai.memory) >= main_white_ai.batch_size: # Train white AI too
                main_white_ai.train() # No need to track loss for both for now

        # Update target networks and schedulers in main process
        main_black_ai.soft_update_target_model(tau=0.001)
        main_white_ai.soft_update_target_model(tau=0.001)
        main_black_ai.scheduler.step() # Step scheduler per episode group
        main_white_ai.scheduler.step() # Step scheduler per episode group

        # Update main stats
        stats['episode_rewards'].extend(group_episode_rewards)
        stats['black_wins'] += group_black_wins
        stats['white_wins'] += group_white_wins
        stats['draws'] += group_draws
        if episode_losses:
            stats['avg_loss'].append(np.mean(episode_losses))
        if group_q_values:
            stats['avg_q_value'].append(np.mean(group_q_values))


        # Print progress
        current_episode = (episode_group + 1) * num_workers
        elapsed_minutes = (time.time() - start_time) / 60
        avg_loss = np.mean(stats['avg_loss'][-10:]) if stats['avg_loss'] else 0
        avg_q = np.mean(stats['avg_q_value'][-10:]) if stats['avg_q_value'] else 0
        lr = main_black_ai.optimizer.param_groups[0]['lr']

        print(f"Episode Group {episode_group+1}/{num_episodes // num_workers} (Episodes {current_episode}/{num_episodes}, {elapsed_minutes:.1f} min) | "
              f"Black/White/Draw: {stats['black_wins']}/{stats['white_wins']}/{stats['draws']} | "
              f"Epsilon: {main_black_ai.epsilon:.3f} | Avg Loss: {avg_loss:.4f} | Avg Q: {avg_q:.4f} | LR: {lr:.6f}")


        if current_episode % save_interval == 0:
            main_black_ai.save_model(f"black_ai_episode_{current_episode}.pth")
            main_white_ai.save_model(f"white_ai_episode_{current_episode}.pth")

    # Signal experience collector to stop (optional, queue timeout will handle it anyway)
    # experience_queue.put(None) # Or a sentinel value if needed for more controlled shutdown
    collector_process.join() # Wait for collector to finish

    # Save final models
    main_black_ai.save_model("black_ai_final.pth")
    main_white_ai.save_model("white_ai_final.pth")

    total_time = (time.time() - start_time) / 60
    print(f"Total training time: {total_time:.2f} minutes")

    return main_black_ai, main_white_ai, stats


if __name__ == '__main__':
    # Set multiprocessing start method to 'spawn' - IMPORTANT for CUDA
    mp.set_start_method('spawn', force=True) # Add this at the very beginning of __main__

    num_processes = 4 # Set number of parallel processes
    black_ai, white_ai, stats = train_ai_parallel(num_episodes=2000, time_limit_minutes=120, num_workers=num_processes)