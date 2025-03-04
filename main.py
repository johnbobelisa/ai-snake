# -----------------------------------------
# Reworked code with the same logic, just different names, structure, 
# added background image support, and snake eyes.
# -----------------------------------------

import pygame
import neat
import numpy as np
import pickle
import argparse
from pygame.locals import (
    K_w, K_a, K_s, K_d,  
    KEYDOWN, QUIT, K_ESCAPE
)

# -----------------------------------------
# Global constants and settings (renamed)
# -----------------------------------------
CELL_SIZE = 25          # Size of each grid cell in pixels
GRID_DIMENSION = 25     # Number of cells per side of the grid
WINDOW_SIZE = CELL_SIZE * GRID_DIMENSION

IS_VISUAL = False            # If True, the game is rendered
LIMIT_FPS = False            # If True, frames are limited to GAME_FPS
GAME_FPS = 10                # Default frames per second
PRINT_DEATH_REASON = False   # Print cause of snake death when True

ROUNDS_PER_AGENT = 3  # Times each AI is tested to get average fitness
VISION_FIELD = 5      # The NxN area around the snake head used as input

# Add background image path (None means solid color if no custom image provided)
BACKGROUND_IMG_PATH = "bg.png"
BACKGROUND_IMG = None  # Will hold the loaded background image, if any

# -----------------------------------------
# Helper functions (some renamed)
# -----------------------------------------
def random_location():
    """Generate a random (x, y) location within the grid."""
    return np.random.randint(0, GRID_DIMENSION, size=2)

def is_wall_or_snake(pos, snake):
    """
    Return 1 if the position is occupied by the snake's body or 
    is outside the grid (like a wall). Otherwise return 0.
    """
    if pos in snake.segments:
        return 1
    if not (0 <= pos[0] < GRID_DIMENSION and 0 <= pos[1] < GRID_DIMENSION):
        return 1
    return 0

def sense_environment(snake, candy):
    """
    Build the local state of the game for NEAT's neural net input.
    The area is a VISION_FIELD x VISION_FIELD box around the snake head.
    """
    inputs = []
    head = snake.segments[0]

    # Mark grid cells (occupied or not) around the head
    for row_offset in range(-VISION_FIELD // 2 + 1, VISION_FIELD // 2 + 1):
        for col_offset in range(-VISION_FIELD // 2 + 1, VISION_FIELD // 2 + 1):
            if row_offset != 0 or col_offset != 0:
                check_pos = (head[0] + col_offset, head[1] + row_offset)
                inputs.append(is_wall_or_snake(check_pos, snake))

    # Food direction (up, down, left, right) relative to head
    food_dir = (
        1 if candy.location[1] < head[1] else 0,  # Up
        1 if candy.location[1] > head[1] else 0,  # Down
        1 if candy.location[0] < head[0] else 0,  # Left
        1 if candy.location[0] > head[0] else 0   # Right
    )
    inputs.extend(food_dir)
    return inputs

def human_control(snake, candy):
    """Control the snake with WASD keys for human play."""
    keys = pygame.key.get_pressed()
    if keys[K_w]:
        snake.direction = snake.move_up
    elif keys[K_s]:
        snake.direction = snake.move_down
    elif keys[K_a]:
        snake.direction = snake.move_left
    elif keys[K_d]:
        snake.direction = snake.move_right

def ai_controller_builder(neat_model):
    """
    Create an AI-based mover function using the provided NEAT model.
    This function is returned and used to control the snake in the game.
    """
    def ai_mover(snake, candy):
        state = sense_environment(snake, candy)
        outputs = neat_model.activate(state)
        choice = np.argmax(outputs)
        if choice == 0:
            snake.direction = snake.move_up
        elif choice == 1:
            snake.direction = snake.move_right
        elif choice == 2:
            snake.direction = snake.move_down
        elif choice == 3:
            snake.direction = snake.move_left
    return ai_mover

# -----------------------------------------
# Classes (renamed and restructured)
# -----------------------------------------
class Candy:
    """
    Represents the candy (food) in the game.
    Responsible for drawing itself and respawning at a new location.
    """
    def __init__(self, pos):
        self.location = tuple(pos)

    def draw_me(self, display):
        """
        Draw the candy with an inner and outer rectangle to give a border effect.
        """
        border_thickness = 5
        pixel_loc = np.multiply(self.location, CELL_SIZE)
        # Outer rectangle
        pygame.draw.rect(display, (150, 0, 0), (*pixel_loc, CELL_SIZE, CELL_SIZE))
        # Inner rectangle
        pygame.draw.rect(
            display, (255, 0, 0),
            (pixel_loc[0] + border_thickness,
             pixel_loc[1] + border_thickness,
             CELL_SIZE - 2*border_thickness,
             CELL_SIZE - 2*border_thickness)
        )

    def relocate(self, finder_func):
        """
        Respawn the candy at a new location, avoiding the snake's current positions.
        """
        new_spot = tuple(finder_func())
        self.location = new_spot

class Snake:
    """
    The main playable entity (snake), holding segments and direction.
    Responsible for drawing itself and moving.
    """
    def __init__(self, start_pos):
        self.segments = [tuple(start_pos)]
        self.direction = self.move_right  # Default movement direction

    def draw_me(self, display):
        """
        Draw all segments of the snake. The head gets eyes.
        """
        border_thickness = 5
        for idx, seg_pos in enumerate(self.segments):
            pixel_loc = np.multiply(seg_pos, CELL_SIZE)
            # Outer rectangle
            pygame.draw.rect(display, (0, 150, 0), (*pixel_loc, CELL_SIZE, CELL_SIZE))
            # Inner rectangle
            pygame.draw.rect(
                display, (0, 255, 0),
                (pixel_loc[0] + border_thickness,
                 pixel_loc[1] + border_thickness,
                 CELL_SIZE - 2*border_thickness,
                 CELL_SIZE - 2*border_thickness)
            )
            # If this segment is the head, draw eyes
            if idx == 0:  
                eye_radius = 3
                left_eye_center = (pixel_loc[0] + 10, pixel_loc[1] + 10)
                right_eye_center = (pixel_loc[0] + 20, pixel_loc[1] + 10)
                pygame.draw.circle(display, (255, 255, 255), left_eye_center, eye_radius)
                pygame.draw.circle(display, (255, 255, 255), right_eye_center, eye_radius)

    def move_me(self):
        """
        Move the snake in the current direction.
        The head position is recalculated, inserted at the front,
        and the tail is removed.
        """
        new_head = self.direction(self.segments[0])
        self.segments.insert(0, new_head)
        self.segments.pop()

    # Direction methods
    def move_left(self, head):
        return (head[0] - 1, head[1])

    def move_right(self, head):
        return (head[0] + 1, head[1])

    def move_up(self, head):
        return (head[0], head[1] - 1)

    def move_down(self, head):
        return (head[0], head[1] + 1)

# -----------------------------------------
# Game loop (renamed)
# -----------------------------------------
def run_game(snake_controller, candy_controller):
    """
    Run one game of Snake (Snake) using the provided
    snake_controller and candy_controller. Returns the score.
    """
    global IS_VISUAL, LIMIT_FPS, GAME_FPS, PRINT_DEATH_REASON

    pygame.init()
    screen = pygame.display.set_mode([WINDOW_SIZE, WINDOW_SIZE])
    clock = pygame.time.Clock()

    # Possibly load background image if a path is set (and hasn't been loaded yet)
    global BACKGROUND_IMG
    if BACKGROUND_IMG_PATH and BACKGROUND_IMG is None:
        BACKGROUND_IMG = pygame.image.load(BACKGROUND_IMG_PATH).convert()
        BACKGROUND_IMG = pygame.transform.scale(BACKGROUND_IMG, (WINDOW_SIZE, WINDOW_SIZE))

    # Create snake and candy
    snake = Snake(random_location())
    candy = Candy(candy_controller())

    points = 0
    stepsSinceLastFood = 0
    gameover = False

    while not gameover:
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                gameover = True
            elif event.type == QUIT:
                gameover = True

        # Draw background
        if IS_VISUAL:
            if BACKGROUND_IMG:
                screen.blit(BACKGROUND_IMG, (0, 0))
            else:
                screen.fill((0, 0, 0))

            # Draw entities
            snake.draw_me(screen)
            candy.draw_me(screen)
            pygame.display.flip()

        # Timeout check
        if stepsSinceLastFood > 100:
            if PRINT_DEATH_REASON:
                print('timeout')
            break

        # Border collision check
        head_x, head_y = snake.segments[0]
        if not (0 <= head_x < GRID_DIMENSION and 0 <= head_y < GRID_DIMENSION):
            if PRINT_DEATH_REASON:
                print('out map')
            break

        # AI or human control
        snake_controller(snake, candy)

        # Self collision check
        if snake.segments[0] in snake.segments[1:]:
            if PRINT_DEATH_REASON:
                print('hit snake')
            break

        # Candy eaten check
        if snake.segments[0] == candy.location:
            stepsSinceLastFood = 0
            points += 1
            candy.relocate(candy_controller)
            # Avoid spawning on the snake
            while candy.location in snake.segments:
                candy.relocate(candy_controller)
            # Grow snake
            snake.segments.append(snake.segments[0])

        # Move snake
        snake.move_me()

        if LIMIT_FPS:
            clock.tick(GAME_FPS)

        stepsSinceLastFood += 1

    return points

# -----------------------------------------
# NEAT training functions (renamed)
# -----------------------------------------
def evaluate_generation(genomes, neat_config):
    """
    Evaluate fitness of each genome in the current NEAT generation.
    Each genome is tested ROUNDS_PER_AGENT times; average is assigned as fitness.
    """
    for genome_id, genome_data in genomes:
        genome_data.fitness = 0
        brain = neat.nn.FeedForwardNetwork.create(genome_data, neat_config)
        ai_mover = ai_controller_builder(brain)
        for _ in range(ROUNDS_PER_AGENT):
            genome_data.fitness += run_game(ai_mover, random_location) / ROUNDS_PER_AGENT

def run_training(config_path, visualize_training=False):
    """
    Train the NEAT-based snake using the specified config file.
    Optionally show the training visually (slower).
    """
    global IS_VISUAL, LIMIT_FPS, GAME_FPS
    if visualize_training:
        IS_VISUAL = True
        LIMIT_FPS = True
        GAME_FPS = 100
    else:
        IS_VISUAL = False
        LIMIT_FPS = False

    neat_config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(neat_config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(50))

    # Up to 1000 generations
    winner_genome = population.run(evaluate_generation, 1000)

    pickle.dump(winner_genome, open('best_genome.pkl', 'wb'))
    print('\nBest genome:\n{!s}'.format(winner_genome))

# -----------------------------------------
# Human play and watchers (renamed)
# -----------------------------------------
def human_play():
    """
    Let a human control the snake with WASD keys.
    """
    global IS_VISUAL, LIMIT_FPS, GAME_FPS
    IS_VISUAL = True
    LIMIT_FPS = True
    GAME_FPS = 10

    final_score = run_game(human_control, random_location)
    print(f"Score: {final_score}")

def preset_candy_positions_maker(positions):
    """
    Create a function that returns candy positions from a preset list.
    Once the list is exhausted, it falls back to random locations.
    """
    pos_list = positions[:]
    def get_next():
        try:
            return pos_list.pop(0)
        except IndexError:
            print('out of given positions; using random ones')
            return random_location()
    return get_next

def visualize_bot(genome_path, config_path, candy_positions_path=None):
    """
    Watch a trained genome control the snake in the game.
    Optionally use a preset list of candy positions.
    """
    global IS_VISUAL, LIMIT_FPS, GAME_FPS
    IS_VISUAL = True
    LIMIT_FPS = True
    GAME_FPS = 30

    trained_genome = pickle.load(open(genome_path, 'rb'))
    neat_config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )
    brain = neat.nn.FeedForwardNetwork.create(trained_genome, neat_config)
    snake_controller = ai_controller_builder(brain)

    if candy_positions_path:
        positions_list = pickle.load(open(candy_positions_path, 'rb'))
        candy_controller = preset_candy_positions_maker(positions_list)
    else:
        candy_controller = random_location

    final_score = run_game(snake_controller, candy_controller)
    print('Score:', final_score)

# -----------------------------------------
# Entry point (renamed, but argument logic the same)
# -----------------------------------------
def entry_point():
    parser = argparse.ArgumentParser(description="Snake Game with NEAT Bot (Reworked)")
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')

    # Train mode
    train_sub = subparsers.add_parser('train', help='Train the bot using NEAT')
    train_sub.add_argument('--config', required=True, help='Path to NEAT config file')
    train_sub.add_argument('--watch', action='store_true', help='Watch the training process')

    # Human play mode
    play_sub = subparsers.add_parser('play', help='Play Snake as a human')

    # Watch mode
    watch_sub = subparsers.add_parser('watch', help='Watch the bot play Snake')
    watch_sub.add_argument('--genome', required=True, help='Path to genome file')
    watch_sub.add_argument('--config', required=True, help='Path to NEAT config file')
    watch_sub.add_argument('--food-pos', help='Path to food positions file (optional)')

    args = parser.parse_args()

    if args.mode == 'train':
        run_training(args.config, args.watch)
    elif args.mode == 'play':
        human_play()
    elif args.mode == 'watch':
        visualize_bot(args.genome, args.config, args.food_pos)
    else:
        parser.print_help()

if __name__ == '__main__':
    entry_point()
