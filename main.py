# main.py

from abalone_game import AbaloneGame, Player
from abalone_ai import AbaloneAI
from training import train_ai
from evaluation import evaluate_ai

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
            print(
                f"Models loaded successfully. Black AI at step {black_ai.steps}, White AI at step {white_ai.steps}")
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
            black_save_path = input(
                "Enter path for saving black AI model (leave empty for default 'black_ai_final.pth'): ")
            white_save_path = input(
                "Enter path for saving white AI model (leave empty for default 'white_ai_final.pth'): ")

            black_save_path = black_save_path or "black_ai_final.pth"
            white_save_path = white_save_path or "white_ai_final.pth"

            black_ai.save_model(black_save_path)
            white_ai.save_model(white_save_path)
            print(f"Models saved to {black_save_path} and {white_save_path}")


if __name__ == "__main__":
    main()