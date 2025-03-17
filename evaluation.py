# evaluation.py

import time
import numpy as np

from abalone_game import AbaloneGame, Player
from abalone_ai import AbaloneAI

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
            center_positions = [(-1, 1, 0), (0, 1, -1), (1, 1, -2),
            (-1, 0, 1), (0, 0, 0), (1, 0, -1),
            (-1, -1, 2), (0, -1, 1), (1, -1, 0)
        ]  #Cube Coordinates
            black_center = sum(1 for pos in center_positions if game.is_valid_position(pos) and game.board.get(pos) == black_ai.player)
            white_center = sum(1 for pos in center_positions if game.is_valid_position(pos) and game.board.get(pos) == white_ai.player)
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
        stats['center_control']['black'].append(
            game_center_control['black'] / game.turn_count if game.turn_count > 0 else 0)
        stats['center_control']['white'].append(
            game_center_control['white'] / game.turn_count if game.turn_count > 0 else 0)

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
# evaluation.py (continued)

    stats['avg_win_margin'] = np.mean(stats['win_margin']) if stats['win_margin'] else 0

    # Print detailed evaluation results
    print("\nDetailed Evaluation Results:")
    print(f"Games played: {num_games}")
    print(f"Black wins: {stats['black_wins']} ({stats['black_wins'] / num_games * 100:.1f}%)")
    print(f"White wins: {stats['white_wins']} ({stats['white_wins'] / num_games * 100:.1f}%)")
    print(f"Draws: {stats['draws']} ({stats['draws'] / num_games * 100:.1f}%)")
    print(f"Average game length: {stats['avg_game_length']:.1f} turns")
    print(f"Average marbles pushed off - Black: {stats['pushed_marbles']['black'] / num_games:.2f}, "
          f"White: {stats['pushed_marbles']['white'] / num_games:.2f}")
    print(f"Average center control - Black: {stats['center_control']['black']:.2f}, "
          f"White: {stats['center_control']['white']:.2f}")
    if stats['win_margin']:
        print(f"Average win margin: {stats['avg_win_margin']:.2f} marbles")

    return stats