"""
Code for training a Snake bot using the NEAT algorithm.

Major Fixes:
  1. Slowed down training when watch=True by introducing a frame-rate limit 
     inside play_ai().
  2. Called game_instance.show_game_over() when watch=True so that the 
     window remains visible long enough to see what happened at the end of each game.

Use the --no-watch flag to disable visuals if you need faster training.
"""

import pickle
import neat
import numpy as np
import argparse
import pygame
import game  # This imports your game.py module

# Training configuration constants
PLAYS_PER_BOT = 3         # Number of simulations per genome (averaged)
VISION_BOX = 5            # 5x5 local grid (inputs: 5*5 - 1 + 4 = 28)
MAX_STEPS = 500           # Maximum steps allowed per simulation
MAX_NO_FOOD_STEPS = 100   # Maximum consecutive steps without eating

# Global variable to track the best genome so far (for checkpointing)
best_fitness = -float('inf')

def bot_mover_maker(model):
    """
    Given a NEAT model, returns a function that controls the snake's direction.
    The returned function accepts the current snake and food objects.
    """
    def bot_mover(snake, food):
        state = local_state(snake, food)
        output = model.activate(state)
        move_index = np.argmax(output)
        # Map output index to a directional tuple:
        # 0: up, 1: right, 2: down, 3: left.
        if move_index == 0:
            snake.change_direction((0, -1))
        elif move_index == 1:
            snake.change_direction((1, 0))
        elif move_index == 2:
            snake.change_direction((0, 1))
        elif move_index == 3:
            snake.change_direction((-1, 0))
    return bot_mover

def is_occupied(pos, snake):
    """
    Returns 1 if the grid cell 'pos' is occupied by the snake or is out-of-bounds;
    otherwise returns 0.
    """
    if not (0 <= pos[0] < game.GRID_SIZE and 0 <= pos[1] < game.GRID_SIZE):
        return 1
    if pos in snake.body:
        return 1
    return 0

def local_state(snake, food):
    """
    Returns a flattened binary vector representing:
      - The VISION_BOX x VISION_BOX local grid (excluding the head).
      - Four booleans indicating if the food is above, below, left, or right of the head.
      
    Total inputs = (VISION_BOX^2 - 1) + 4.
    """
    state = []
    head = snake.body[0]
    half = VISION_BOX // 2
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            if i == 0 and j == 0:
                continue  # Skip the head cell
            cell = (head[0] + j, head[1] + i)
            state.append(is_occupied(cell, snake))
    # Append directional booleans for the food position relative to the head.
    state.extend([
        1 if food.position[1] < head[1] else 0,  # food is up
        1 if food.position[1] > head[1] else 0,  # food is down
        1 if food.position[0] < head[0] else 0,  # food is left
        1 if food.position[0] > head[0] else 0   # food is right
    ])
    return state

def play_ai(bot_mover, max_steps=MAX_STEPS, max_no_food_steps=MAX_NO_FOOD_STEPS, watch=False):
    """
    Runs a single game simulation using the provided bot_mover function.
    
    The simulation terminates if:
      - The game ends (collision),
      - The maximum number of steps is reached, or
      - Too many consecutive steps occur without eating food.
    
    If 'watch' is True, we:
      - Render the game each frame.
      - Limit the frame rate (default ~x FPS).
      - Show a "Game Over" screen at the end.
    
    Returns the final score (number of foods eaten) as the fitness measure.
    """
    game_instance = game.Game()
    clock = None
    if watch:
        clock = pygame.time.Clock()

    steps = 0
    no_food_steps = 0
    current_score = game_instance.score

    while (not game_instance.game_over and steps < max_steps 
           and no_food_steps < max_no_food_steps):
        # If watch=True, process events so the window remains responsive.
        if watch:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_instance.game_over = True

        bot_mover(game_instance.snake, game_instance.food)
        game_instance.update()

        if watch:
            game_instance.draw()
            clock.tick(360)  # change fps here

        steps += 1
        if game_instance.score > current_score:
            current_score = game_instance.score
            no_food_steps = 0
        else:
            no_food_steps += 1

    # When the loop ends, optionally show the game over screen.
    if watch:
        game_instance.show_game_over()

    return game_instance.score

def train_generation(genomes, config, watch=False):
    """
    Evaluates each genome by averaging its performance over multiple game simulations.
    Also saves a checkpoint if a new highest fitness is achieved.
    """
    global best_fitness

    for genome_id, genome in genomes:
        genome.fitness = 0
        try:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            mover = bot_mover_maker(net)
            total_score = 0
            for _ in range(PLAYS_PER_BOT):
                total_score += play_ai(mover, watch=watch)
            genome.fitness = total_score / PLAYS_PER_BOT
        except Exception as e:
            print(f"Error evaluating genome {genome_id}: {e}")
            genome.fitness = -1

    # Find best genome in this generation.
    best_genome_in_generation = max(genomes, key=lambda x: x[1].fitness)[1]
    if best_genome_in_generation.fitness > best_fitness:
        best_fitness = best_genome_in_generation.fitness
        with open('best_model_checkpoint.pkl', 'wb') as f:
            pickle.dump(best_genome_in_generation, f)
        print(f"New checkpoint saved with fitness {best_fitness}")

def run_neat(config_file, generations=1000, use_parallel=False, watch=False):
    """
    Sets up and runs the NEAT algorithm using the provided configuration file.
    
    Parameters:
      - config_file: path to the NEAT configuration file.
      - generations: maximum number of generations to run.
      - use_parallel: if True and visuals are disabled, use parallel evaluation.
      - watch: if True, we slow training down (~10 FPS) so you can watch the AI learn.
    """
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_file)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))
    
    # If watch=True, we won't do parallel evaluation (it doesn't make sense to see multiple snakes).
    if use_parallel and not watch:
        import multiprocessing
        def eval_genome(genome, config):
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            mover = bot_mover_maker(net)
            return play_ai(mover, watch=False)  # No visuals in parallel mode
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
        winner = p.run(pe.evaluate, generations)
    else:
        winner = p.run(lambda gs, c: train_generation(gs, c, watch=watch), generations)
    
    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)
    with open('best_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    print("\nBest genome:\n{}".format(winner))

def parse_args():
    """
    Parses command-line arguments.
    Use --no-watch to disable visuals (default is to watch the simulation).
    """
    parser = argparse.ArgumentParser(description="Train a Snake bot using NEAT and watch it learn.")
    parser.add_argument('--config', type=str, default='config-nn.txt',
                        help="Path to the NEAT configuration file.")
    parser.add_argument('--generations', type=int, default=1000,
                        help="Number of generations to run.")
    parser.add_argument('--parallel', action='store_true',
                        help="Use parallel evaluation (disabled if visuals are on).")
    parser.add_argument('--no-watch', dest='watch', action='store_false',
                        help="Disable rendering of the simulation.")
    parser.set_defaults(watch=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_neat(args.config, generations=args.generations, use_parallel=args.parallel, watch=args.watch)
