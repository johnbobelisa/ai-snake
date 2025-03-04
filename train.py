import pickle
import neat
import numpy as np
import argparse
import pygame
import game  # Imports your game.py module

# Training configuration constants
PLAYS_PER_BOT = 3         # Number of simulations per genome (averaged)
MAX_STEPS = 500           # Maximum steps allowed per simulation
MAX_NO_FOOD_STEPS = 100   # Maximum consecutive steps without eating

# Global variable to track the best genome so far (for checkpointing)
best_fitness = -float('inf')

def bot_mover_maker(model):
    """
    Returns a function that controls the snake's direction based on the NEAT model.
    """
    DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
    
    def bot_mover(snake, food):
        state = local_state(snake, food)
        output = model.activate(state)  # [turn_left, go_straight, turn_right]
        action_index = np.argmax(output)
        
        current_index = DIRECTIONS.index(snake.direction)
        if action_index == 0:  # turn_left
            new_index = (current_index - 1) % 4
        elif action_index == 1:  # go_straight
            new_index = current_index
        elif action_index == 2:  # turn_right
            new_index = (current_index + 1) % 4
        
        snake.change_direction(DIRECTIONS[new_index])
    
    return bot_mover

def local_state(snake, food):
    """
    Returns a 28-element state vector for the NEAT network using ray tracing:
    - 8 directions * 3 sensors (wall, body, food) = 24 inputs
    - 4 inputs for one-hot encoded heading = 4 inputs
    """
    state = []
    head = snake.body[0]
    snake_set = set(snake.body)  # O(1) lookups
    
    ray_directions = [
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1)  # N, NE, E, SE, S, SW, W, NW
    ]
    
    for dx, dy in ray_directions:
        distance_to_wall = None
        distance_to_snake_body = None
        distance_to_food = None
        
        k = 1
        while True:
            pos = (head[0] + k * dx, head[1] + k * dy)
            if not (0 <= pos[0] < game.GRID_SIZE and 0 <= pos[1] < game.GRID_SIZE):
                distance_to_wall = k
                break
            if pos in snake_set and distance_to_snake_body is None:
                distance_to_snake_body = k
            if pos == food.position and distance_to_food is None:
                distance_to_food = k
            k += 1
        
        wall_sensor = 1 / (distance_to_wall + 1)
        body_sensor = 1 / (distance_to_snake_body + 1) if distance_to_snake_body else 0
        food_sensor = 1 / (distance_to_food + 1) if distance_to_food else 0
        state.extend([wall_sensor, body_sensor, food_sensor])
    
    heading_directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    heading_index = heading_directions.index(snake.direction)
    heading_one_hot = [0] * 4
    heading_one_hot[heading_index] = 1
    state.extend(heading_one_hot)
    
    return state

def play_ai(bot_mover, max_steps=MAX_STEPS, max_no_food_steps=MAX_NO_FOOD_STEPS, watch=False):
    """
    Runs a single game simulation with the given bot_mover.
    Returns the score (foods eaten).
    """
    game_instance = game.Game()
    clock = pygame.time.Clock() if watch else None

    steps = 0
    no_food_steps = 0
    current_score = game_instance.score

    while (not game_instance.game_over and steps < max_steps 
           and no_food_steps < max_no_food_steps):
        if watch:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_instance.game_over = True

        bot_mover(game_instance.snake, game_instance.food)
        game_instance.update()

        if watch:
            game_instance.draw()
            clock.tick(10)  # 10 FPS when watching

        steps += 1
        if game_instance.score > current_score:
            current_score = game_instance.score
            no_food_steps = 0
        else:
            no_food_steps += 1

    if watch:
        game_instance.show_game_over()

    return game_instance.score

def train_generation(genomes, config, watch=False):
    """
    Evaluates genomes sequentially, averaging scores over PLAYS_PER_BOT runs.
    Tracks and prints the best single-run score per generation.
    """
    global best_fitness
    generation_best_max_score = 0

    for genome_id, genome in genomes:
        genome.fitness = 0
        try:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            mover = bot_mover_maker(net)
            total_score = 0
            genome_max_score = 0
            for _ in range(PLAYS_PER_BOT):
                score = play_ai(mover, watch=watch)
                total_score += score
                genome_max_score = max(genome_max_score, score)
            genome.fitness = total_score / PLAYS_PER_BOT
            generation_best_max_score = max(generation_best_max_score, genome_max_score)
        except Exception as e:
            print(f"Error evaluating genome {genome_id}: {e}")
            genome.fitness = -1

    best_genome = max(genomes, key=lambda x: x[1].fitness)[1]
    if best_genome.fitness > best_fitness:
        best_fitness = best_genome.fitness
        with open('best_model_checkpoint.pkl', 'wb') as f:
            pickle.dump(best_genome, f)
        print(f"New checkpoint saved with fitness {best_fitness}")

    print(f"Generation best single-run score: {generation_best_max_score}")

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    mover = bot_mover_maker(net)
    return play_ai(mover, watch=False)


def run_neat(config_file, generations=1000, use_parallel=False, watch=False):
    """
    Runs the NEAT algorithm with the specified settings.
    """
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(50))
    
    if use_parallel and not watch:
        from multiprocessing import cpu_count
        pe = neat.ParallelEvaluator(cpu_count(), eval_genome)
        winner = p.run(pe.evaluate, generations)
    else:
        winner = p.run(lambda gs, c: train_generation(gs, c, watch=watch), generations)
    
    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)
    with open('best_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    print(f"\nBest genome:\n{winner}")

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Snake bot with NEAT.")
    parser.add_argument('--config', type=str, default='config-nn.txt')
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--no-watch', dest='watch', action='store_false')
    parser.set_defaults(watch=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_neat(args.config, args.generations, args.parallel, args.watch)
