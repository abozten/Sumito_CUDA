import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Set, Optional

class Player(Enum):
    BLACK = 1
    WHITE = 2
    
    def opponent(self):
        return Player.WHITE if self == Player.BLACK else Player.BLACK

class AbaloneGame:
    """
    Implementation of the Abalone board game for AI training.
    
    The board uses cube coordinates for the hexagonal grid:
    - Each position is represented as (q, r, s) where q + r + s = 0
    - Six movement directions are possible
    """
    
    # Six directions in cube coordinates
    DIRECTIONS = [
        (+1, -1, 0), (+1, 0, -1), (0, +1, -1),
        (-1, +1, 0), (-1, 0, +1), (0, -1, +1)
    ]
    
    def __init__(self):
        # Initialize the board
        self.reset()
    
    def reset(self):
        """Reset the game to the initial state."""
        self.current_player = Player.BLACK
        self.turn_count = 1
        
        # Create an empty board
        self.board = {}
        
        # Define the board radius (distance from center to edge)
        radius = 4
        
        # Generate all valid board positions
        for q in range(-radius, radius+1):
            for r in range(-radius, radius+1):
                s = -q-r
                if abs(s) <= radius:
                    self.board[(q, r, s)] = None
        
        # Place the initial marbles according to the standard layout
        # White marbles in the top
        white_positions = [
            # row 1 => 5 white
            (-4, 0, 4), (-3, 0, 3), (-2, 0, 2), (-1, 0, 1), (0, 0, 0),
            # row 2 => 6 white
            (-4, 1, 3), (-3, 1, 2), (-2, 1, 1), (-4, 1, 0), (-4, 1, -1), (1, 1, -2),
            # row 3 => 3 white
            (-2, 2, 0), (-1, 2, -1), (0, 2, -2)
        ]
        
        # Black marbles in the bottom
        black_positions = [
            # row 1 => 5 black
            (0, 0, 1), (1, 0, 0), (2, 0, -1), (3, 0, -2), (4, 0, -3),
            # row 2 => 6 black
            (-1, -2, 3), (0, -2, 2), (1, -2, 1), (2, -2, 0), (3, -2, -1), (4, -2, -2),
            # row 3 => 3 black
            (-1, -3, 4), (0, -3, 3), (1, -3, 2)
        ]
        
        # Place the marbles on the board
        for pos in black_positions:
            self.board[pos] = Player.BLACK
            
        for pos in white_positions:
            self.board[pos] = Player.WHITE
        
        # Track pushed-off marbles
        self.pushed_off = {Player.BLACK: 0, Player.WHITE: 0}
        
        # Game state
        self.game_over = False
        self.winner = None
        
    def is_valid_position(self, position: Tuple[int, int, int]) -> bool:
        """Check if the position is on the board."""
        return position in self.board
    
    def get_neighbor(self, position: Tuple[int, int, int], direction: int) -> Tuple[int, int, int]:
        """Get the neighboring position in the given direction."""
        q, r, s = position
        dq, dr, ds = self.DIRECTIONS[direction]
        return (q + dq, r + dr, s + ds)
    
    def get_marble_line(self, start_pos: Tuple[int, int, int], direction: int, 
                        max_length: int = 3) -> List[Tuple[int, int, int]]:
        """Get a line of marbles starting from a position in a direction."""
        line = [start_pos]
        current_pos = start_pos
        
        # Find marbles in the line (up to max_length)
        while len(line) < max_length:
            next_pos = self.get_neighbor(current_pos, direction)
            if not self.is_valid_position(next_pos) or self.board[next_pos] != self.board[start_pos]:
                break
            line.append(next_pos)
            current_pos = next_pos
        
        return line
    
    def get_valid_moves(self) -> List[Tuple[List[Tuple[int, int, int]], int]]:
        """
        Get all valid moves for the current player.
        Returns a list of (marble_line, direction) tuples.
        """
        valid_moves = []
        
        # Find all marbles of the current player
        player_marbles = {pos for pos, player in self.board.items() 
                         if player == self.current_player}
        
        # Check each marble and possible lines
        for pos in player_marbles:
            for direction in range(6):
                # Try forming lines of length 1, 2, and 3
                for length in range(1, 4):
                    # Find line of marbles in this direction
                    line = self.get_marble_line(pos, direction, length)
                    
                    if len(line) < length:
                        continue  # Line is shorter than desired length
                    
                    # Check if the line can move in each direction
                    for move_dir in range(6):
                        if self.is_valid_move(line, move_dir):
                            valid_moves.append((line, move_dir))
        
        return valid_moves
    
    def get_line_strength(self, line: List[Tuple[int, int, int]], direction: int) -> Tuple[int, int]:
        """
        Get the strength of a line pushing in a direction.
        Returns (friendly_count, opponent_count)
        """
        friendly_count = len(line)
        opponent_count = 0
        
        # Check the marbles in the direction
        current_pos = line[-1]  # Last position in the line
        
        # First check if there's an immediately adjacent opponent marble
        # (no gaps allowed in sumito)
        next_pos = self.get_neighbor(current_pos, direction)
        if not self.is_valid_position(next_pos) or self.board[next_pos] is None:
            return friendly_count, opponent_count
        
        if self.board[next_pos] != self.current_player:
            # Found an opponent marble, start counting
            opponent_count = 1
            current_pos = next_pos
            
            # Continue counting subsequent opponent marbles in line
            while True:
                next_pos = self.get_neighbor(current_pos, direction)
                # If off board or empty, we're done
                if not self.is_valid_position(next_pos) or self.board[next_pos] is None:
                    break
                # If it's an opponent marble, count it
                if self.board[next_pos] != self.current_player:
                    opponent_count += 1
                    current_pos = next_pos
                else:
                    # Can't push your own marbles
                    return friendly_count, -1
        else:
            # Can't push your own marbles
            return friendly_count, -1
        
        return friendly_count, opponent_count
    
    def is_valid_move(self, line: List[Tuple[int, int, int]], direction: int) -> bool:
        """Check if moving a line of marbles in a direction is valid."""
        if not line:
            return False
        
        # In-line move (parallel to the line)
        if len(line) > 1:
            line_dir = self.get_direction(line[0], line[1])
            
            # Check if moving along the line
            if direction == line_dir:  # Forward
                # Check if there's space in front
                front_pos = self.get_neighbor(line[-1], direction)
                return self.is_valid_position(front_pos) and self.board[front_pos] is None
            
            if direction == (line_dir + 3) % 6:  # Backward
                # Check if there's space behind
                back_pos = self.get_neighbor(line[0], direction)
                return self.is_valid_position(back_pos) and self.board[back_pos] is None
        
        # Broadside move (perpendicular to the line)
        if len(line) > 1:
            line_dir = self.get_direction(line[0], line[1])
            if direction != line_dir and direction != (line_dir + 3) % 6:
                # Check if all spaces next to the line are empty
                for pos in line:
                    next_pos = self.get_neighbor(pos, direction)
                    if not self.is_valid_position(next_pos) or self.board[next_pos] is not None:
                        return False
                return True
        
        # Single marble or sumito (pushing)
        friendly_count, opponent_count = self.get_line_strength(line, direction)
        
        # Check pushing rules
        if opponent_count > 0:
            # Can only push if we have more marbles
            if friendly_count <= opponent_count:
                return False
            
            # Maximum 3 marbles can push
            if friendly_count > 3:
                return False
                
            # According to the rules, only these sumito cases are allowed:
            # 2-to-1, 3-to-1, 3-to-2
            valid_sumito = (
                (friendly_count == 2 and opponent_count == 1) or
                (friendly_count == 3 and opponent_count == 1) or
                (friendly_count == 3 and opponent_count == 2)
            )
            if not valid_sumito:
                return False
                
            # Check if the push is valid (enough space or edge)
            push_line = []
            current_pos = line[-1]
            for _ in range(opponent_count):
                next_pos = self.get_neighbor(current_pos, direction)
                push_line.append(next_pos)
                current_pos = next_pos
            
            # Check the next position after the last opponent
            next_pos = self.get_neighbor(current_pos, direction)
            # It should be either off the board or empty
            if self.is_valid_position(next_pos) and self.board[next_pos] is not None:
                return False
                
            return True
        else:
            # Simple move to empty space
            next_pos = self.get_neighbor(line[-1], direction)
            return self.is_valid_position(next_pos) and self.board[next_pos] is None
    
    def get_direction(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> int:
        """Determine the direction from pos1 to pos2."""
        q1, r1, s1 = pos1
        q2, r2, s2 = pos2
        dq, dr, ds = q2 - q1, r2 - r1, s2 - s1
        
        for i, (dq_dir, dr_dir, ds_dir) in enumerate(self.DIRECTIONS):
            if dq == dq_dir and dr == dr_dir and ds == ds_dir:
                return i
        
        # Not adjacent positions
        return -1
    
    def make_move(self, line: List[Tuple[int, int, int]], direction: int) -> bool:
        """
        Make a move with a line of marbles in a given direction.
        Returns True if the move was successful.
        """
        if not self.is_valid_move(line, direction):
            return False
        
        # Move the marbles
        if len(line) == 1 or direction != self.get_direction(line[0], line[1]) and direction != (self.get_direction(line[0], line[1]) + 3) % 6:
            # Broadside move or single marble move
            new_positions = []
            
            # First, check all destination positions
            for pos in line:
                new_pos = self.get_neighbor(pos, direction)
                new_positions.append(new_pos)
            
            # Then, move all marbles
            for old_pos, new_pos in zip(line, new_positions):
                self.board[new_pos] = self.board[old_pos]
                self.board[old_pos] = None
        else:
            # In-line move
            if direction == self.get_direction(line[0], line[1]):  # Forward
                # Move marbles one by one from front to back
                marbles_to_move = list(reversed(line))
                for pos in marbles_to_move:
                    new_pos = self.get_neighbor(pos, direction)
                    # If moving off the board, count it as pushed off
                    if not self.is_valid_position(new_pos):
                        opponent = self.current_player.opponent()
                        self.pushed_off[opponent] += 1
                    else:
                        self.board[new_pos] = self.board[pos]
                    self.board[pos] = None
            else:  # Backward
                # Move marbles one by one from back to front
                for pos in line:
                    new_pos = self.get_neighbor(pos, direction)
                    # If moving off the board, count it as pushed off
                    if not self.is_valid_position(new_pos):
                        opponent = self.current_player.opponent()
                        self.pushed_off[opponent] += 1
                    else:
                        self.board[new_pos] = self.board[pos]
                    self.board[pos] = None
        
        # Check for game over condition
        if self.pushed_off[Player.BLACK] >= 6:
            self.game_over = True
            self.winner = Player.WHITE
        elif self.pushed_off[Player.WHITE] >= 6:
            self.game_over = True
            self.winner = Player.BLACK
        
        # Switch player
        self.current_player = self.current_player.opponent()
        self.turn_count += 1
        
        return True
    
    def get_state_representation(self) -> np.ndarray:
        """
        Get a matrix representation of the board state for AI training.
        Returns a 3D array where:
        - First channel: 1 for black marbles, 0 otherwise
        - Second channel: 1 for white marbles, 0 otherwise
        - Third channel: 1 if current player is black, 0 if white
        """
        # Find the bounds of the board
        board_size = 9  # Standard Abalone board is 9x9 in a hexagonal pattern
        
        # Create a (3, board_size, board_size) array
        state = np.zeros((3, board_size, board_size), dtype=np.float32)
        
        # Map cube coordinates to 2D array indices
        for pos, player in self.board.items():
            if player is not None:
                q, r, _ = pos
                # Convert cube coordinates to 2D array indices
                i, j = q + 4, r + 4  # +4 to shift to 0-indexed array
                
                if 0 <= i < board_size and 0 <= j < board_size:
                    if player == Player.BLACK:
                        state[0, i, j] = 1
                    else:
                        state[1, i, j] = 1
        
        # Set current player channel
        state[2, :, :] = 1 if self.current_player == Player.BLACK else 0
        
        return state
    
    def display(self):
        """Display the current board state."""
        # Find the bounds of the board
        min_q = min(pos[0] for pos in self.board.keys())
        max_q = max(pos[0] for pos in self.board.keys())
        min_r = min(pos[1] for pos in self.board.keys())
        max_r = max(pos[1] for pos in self.board.keys())
        
        print(f"Turn: {self.turn_count}, Player: {'BLACK' if self.current_player == Player.BLACK else 'WHITE'}")
        print(f"Pushed off - BLACK: {self.pushed_off[Player.BLACK]}, WHITE: {self.pushed_off[Player.WHITE]}")
        
        # Print the board
        for r in range(min_r, max_r + 1):
            # Add indentation based on row to align the hexagons
            indent = " " * (r - min_r)
            row = indent
            
            for q in range(min_q, max_q + 1):
                s = -q - r
                pos = (q, r, s)
                
                if pos in self.board:
                    if self.board[pos] == Player.BLACK:
                        row += "B "
                    elif self.board[pos] == Player.WHITE:
                        row += "W "
                    else:
                        row += ". "
                else:
                    row += "  "
            
            print(row)

# Example usage for AI training

def main():
    game = AbaloneGame()
    game.display()
    input("Press Enter to start the game...")
    
    # Game loop
    while not game.game_over:
        game.display()
        
        # Get valid moves
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            print("No valid moves available. Game ends in a draw.")
            break
        
        # Display current player
        print(f"\nCurrent player: {game.current_player.name}")
        
        # Ask the user for input or use random move
        move_choice = input("Select [r] for random move or [m] to choose a move manually: ").lower().strip()
        
        if move_choice == 'r':
            # Use random move
            import random
            move = random.choice(valid_moves)
            line, direction = move
            print(f"Selected random move: Line {line} in direction {direction}")
        else:
            # Display available moves
            print("\nAvailable moves:")
            for i, (line, direction) in enumerate(valid_moves):
                dir_names = ["NE", "E", "SE", "SW", "W", "NW"]
                print(f"{i}: Move line {line} in direction {dir_names[direction]}")
            
            # Get user selection
            while True:
                try:
                    move_index = int(input(f"Select move (0-{len(valid_moves)-1}): "))
                    if 0 <= move_index < len(valid_moves):
                        break
                    else:
                        print(f"Please enter a number between 0 and {len(valid_moves)-1}")
                except ValueError:
                    print("Please enter a valid number")
            
            # Make the selected move
            move = valid_moves[move_index]
            line, direction = move
            print(f"Selected move: Line {line} in direction {direction}")
        
        # Make the move
        game.make_move(line, direction)
    
    # Game over
    game.display()
    if game.winner:
        print(f"Game over! {game.winner.name} wins!")
    else:
        print("Game over! It's a draw!")

if __name__ == "__main__":
    main()