"""
Code to watch the best bot play Snake.

Major Fix:
  - We now respect the same 'clock.tick(10)' limit in play_ai(), ensuring 
    the game doesn't end instantly on a fast CPU. Also show the "Game Over"
    screen so the window doesn't vanish immediately.
"""

import pickle
import pygame
import neat
import game  # Your game module
from train import bot_mover_maker, play_ai  # Reuse helper functions from train.py

def preset_food_pos_maker(positions):
    """
    Returns a function that returns the next position in 'positions' each time it's called.
    If the list is exhausted, returns a random position.
    """
    pos = positions.copy()
    def preset_food_pos():
        try:
            return pos.pop(0)
        except Exception:
            print('Out of given positions; using random ones')
            return (game.GRID_SIZE // 2, game.GRID_SIZE // 2)  # or game.rand_pos()
    return preset_food_pos

def watch_best(genome_file, config_file, food_pos_file=None):
    """
    Plays one game of Snake using the best genome.
    If a file of food positions is provided, it will use them; otherwise, food is placed randomly.
    
    We run the same play_ai() logic as in training, but with watch=True, so the
    game is drawn at ~10 FPS, and the window remains long enough to see the result.
    """
    # Load best genome and configuration (pickled during training)
    with open(genome_file, 'rb') as f:
        genome = pickle.load(f)
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    
    # Create the model from the best genome.
    model = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Force visual mode: Slower updates, show game over, etc.
    game.WATCH = True
    game.USE_FRAMERATE = True
    game.FRAMERATE = 10  # Could reduce further if you'd like it even slower

    # Create snake controller using the model.
    snake_controller = bot_mover_maker(model)
    
    # For a future extension, you might pass a custom food controller into play_ai, 
    # but in our current code, we always place food randomly or in the 
    # next step. The preset_food_pos_maker approach is an example of how you might do so.

    # If you want to incorporate preset food placement into play_ai, 
    # you'd have to adapt the code in train.py or game.py 
    # to accept a function that returns the next food position.

    score = play_ai(snake_controller, watch=True)
    print('Final Score:', score)

def watch_games(genome_file, config_file):
    """
    Repeatedly plays games using the best genome, allowing you to watch the snake 
    multiple times in a row. Each new run starts a fresh game.
    """
    with open(genome_file, 'rb') as f:
        genome = pickle.load(f)
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    
    model = neat.nn.FeedForwardNetwork.create(genome, config)
    
    game.WATCH = True
    game.USE_FRAMERATE = True
    game.FRAMERATE = 60

    snake_controller = bot_mover_maker(model)
    
    while True:
        score = play_ai(snake_controller, watch=True)
        print('Score:', score)
        pygame.time.wait(1500)  # Pause 1.5 seconds between runs

if __name__ == '__main__':
    # Example usage:
    # To watch the best genome once with default random food positions:
    watch_best('best_genome.pkl', 'best_config.pkl')
    
    # If you want to watch repeated games, uncomment:
    # watch_games('best_genome.pkl', 'best_config.pkl')
