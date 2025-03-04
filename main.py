import pygame
import neat
import numpy as np
import pickle
import argparse
from pygame.locals import (
    K_w, K_a, K_s, K_d,
    KEYDOWN, QUIT, K_ESCAPE
)

# ------------------------------------------------
# Global constants and settings (modified)
# ------------------------------------------------
CELL_SIZE = 25          # Size of each grid cell in pixels
GRID_DIMENSION = 25     # Number of cells per side of the grid
WINDOW_SIZE = CELL_SIZE * GRID_DIMENSION

IS_VISUAL = False            # If True, the game is rendered
LIMIT_FPS = False            # If True, frames are limited to GAME_FPS
GAME_FPS = 10                # Default frames per second
PRINT_DEATH_REASON = False   # Print cause of snake death when True

ROUNDS_PER_AGENT = 3  # Times each AI is tested to get average fitness
VISION_FIELD = 7      # Expanded vision field (was 5)

# Background image support
BACKGROUND_IMG_PATH = "bg.png"
BACKGROUND_IMG = None  # Will hold the loaded background image, if any

# ------------------------------------------------
# Helper functions
# ------------------------------------------------
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

def sense_environment(snake, candy, obstacles=None):
    """
    Build the local state for the neural net with three new additions:
      - Expanded vision (VISION_FIELD x VISION_FIELD around the head)
      - Direction memory: includes previous 2 moves (for each direction, 1 if used)
      - Danger proximity sensors: provide inverse distance to danger in each cardinal direction
    If obstacles are provided, they are also treated as dangers.
    """
    inputs = []
    head = snake.segments[0]
    
    # Expanded vision: scan a VISION_FIELD x VISION_FIELD area around the head (exclude head itself)
    for row_offset in range(-VISION_FIELD // 2 + 1, VISION_FIELD // 2 + 1):
        for col_offset in range(-VISION_FIELD // 2 + 1, VISION_FIELD // 2 + 1):
            if row_offset != 0 or col_offset != 0:
                check_pos = (head[0] + col_offset, head[1] + row_offset)
                if obstacles:
                    # Use a combined occupied function to account for obstacles.
                    def is_occupied(pos):
                        return is_wall_or_snake(pos, snake) or pos in obstacles.positions
                    inputs.append(1 if is_occupied(check_pos) else 0)
                else:
                    inputs.append(is_wall_or_snake(check_pos, snake))
    
    # Food direction relative to the head (up, down, left, right)
    food_dir = (
        1 if candy.location[1] < head[1] else 0,  # Up
        1 if candy.location[1] > head[1] else 0,  # Down
        1 if candy.location[0] < head[0] else 0,  # Left
        1 if candy.location[0] > head[0] else 0   # Right
    )
    inputs.extend(food_dir)
    
    # Direction memory: add binary flags for whether each direction was used in the last two moves
    for direction in [snake.move_up, snake.move_right, snake.move_down, snake.move_left]:
        inputs.append(1 if direction in snake.prev_moves else 0)
    
    # Danger proximity sensors: for each direction, check the inverse of distance to a wall, snake, or obstacle
    danger_distances = [0, 0, 0, 0]  # [up, right, down, left]
    for idx, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
        for i in range(1, GRID_DIMENSION):
            check_pos = (head[0] + dx * i, head[1] + dy * i)
            if obstacles:
                def is_occupied(pos):
                    return is_wall_or_snake(pos, snake) or pos in obstacles.positions
            else:
                is_occupied = lambda pos: is_wall_or_snake(pos, snake)
            if is_occupied(check_pos):
                danger_distances[idx] = 1/i  # Inverse distance sensor
                break
    inputs.extend(danger_distances)
    
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
    """
    def ai_mover(snake, candy):
        # For simplicity, we assume no obstacles are provided here.
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

# ------------------------------------------------
# Classes with modifications
# ------------------------------------------------
class Candy:
    """
    Represents the candy (food) with dynamic value that increases over time.
    """
    def __init__(self, pos):
        self.location = tuple(pos)
        self.value = 1            # Initial value
        self.time_on_board = 0    # Time counter for dynamic food
    
    def draw_me(self, display):
        border_thickness = 5
        pixel_loc = np.multiply(self.location, CELL_SIZE)
        pygame.draw.rect(display, (150, 0, 0), (*pixel_loc, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(
            display, (255, 0, 0),
            (pixel_loc[0] + border_thickness,
             pixel_loc[1] + border_thickness,
             CELL_SIZE - 2*border_thickness,
             CELL_SIZE - 2*border_thickness)
        )
    
    def relocate(self, finder_func):
        """Respawn the candy at a new location and reset its dynamic properties."""
        new_spot = tuple(finder_func())
        self.location = new_spot
        self.value = 1
        self.time_on_board = 0

class Snake:
    """
    The snake class now has:
      - A memory of the last two moves (direction memory)
      - A history of positions to penalize redundant movement (path efficiency)
    """
    def __init__(self, start_pos):
        self.segments = [tuple(start_pos)]
        self.direction = self.move_right  # Default movement direction
        self.prev_moves = [self.move_right, self.move_right]  # Store last 2 moves
        self.position_history = set()  # Track visited positions
        self.revisit_count = 0         # Count revisits for path efficiency
    
    def draw_me(self, display):
        border_thickness = 5
        for idx, seg_pos in enumerate(self.segments):
            pixel_loc = np.multiply(seg_pos, CELL_SIZE)
            pygame.draw.rect(display, (0, 150, 0), (*pixel_loc, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(
                display, (0, 255, 0),
                (pixel_loc[0] + border_thickness,
                 pixel_loc[1] + border_thickness,
                 CELL_SIZE - 2*border_thickness,
                 CELL_SIZE - 2*border_thickness)
            )
            if idx == 0:  # Draw eyes for the head
                eye_radius = 3
                left_eye_center = (pixel_loc[0] + 10, pixel_loc[1] + 10)
                right_eye_center = (pixel_loc[0] + 20, pixel_loc[1] + 10)
                pygame.draw.circle(display, (255, 255, 255), left_eye_center, eye_radius)
                pygame.draw.circle(display, (255, 255, 255), right_eye_center, eye_radius)
    
    def move_me(self):
        """
        Move the snake in the current direction.
        Updates the move history and penalizes revisiting positions.
        """
        new_head = self.direction(self.segments[0])
        # Update direction memory: remove oldest, insert current move
        self.prev_moves.pop()
        self.prev_moves.insert(0, self.direction)
        # Path efficiency: penalize if new_head was already visited
        if new_head in self.position_history:
            self.revisit_count += 1
        self.position_history.add(new_head)
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

# New class for static obstacles in the game environment.
class Obstacle:
    def __init__(self, positions):
        self.positions = positions  # List of (x,y) tuples
    
    def draw_me(self, display):
        for pos in self.positions:
            pixel_loc = np.multiply(pos, CELL_SIZE)
            pygame.draw.rect(display, (100, 100, 100), (*pixel_loc, CELL_SIZE, CELL_SIZE))

# ------------------------------------------------
# Game loop with fitness modifications and obstacles
# ------------------------------------------------
def run_game(snake_controller, candy_controller, obstacle_positions=None):
    """
    Run one game instance and return the final score adjusted with:
      - A time efficiency bonus for quick food collection
      - A survival bonus for lasting longer
      - A penalty for redundant movement (path efficiency)
    """
    global IS_VISUAL, LIMIT_FPS, GAME_FPS, PRINT_DEATH_REASON
    pygame.init()
    screen = pygame.display.set_mode([WINDOW_SIZE, WINDOW_SIZE])
    clock = pygame.time.Clock()
    
    global BACKGROUND_IMG
    if BACKGROUND_IMG_PATH and BACKGROUND_IMG is None:
        BACKGROUND_IMG = pygame.image.load(BACKGROUND_IMG_PATH).convert()
        BACKGROUND_IMG = pygame.transform.scale(BACKGROUND_IMG, (WINDOW_SIZE, WINDOW_SIZE))
    
    snake = Snake(random_location())
    candy = Candy(candy_controller())
    
    # Create obstacles if positions are provided
    obstacles = Obstacle(obstacle_positions) if obstacle_positions else Obstacle([])
    
    points = 0
    stepsSinceLastFood = 0
    total_time_bonus = 0  # For time efficiency: bonus for collecting food quickly
    steps_survived = 0    # For survival bonus
    gameover = False
    
    while not gameover:
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                gameover = True
            elif event.type == QUIT:
                gameover = True
        
        # Draw background and game entities if visualization is enabled
        if IS_VISUAL:
            if BACKGROUND_IMG:
                screen.blit(BACKGROUND_IMG, (0, 0))
            else:
                screen.fill((0, 0, 0))
            obstacles.draw_me(screen)
            snake.draw_me(screen)
            candy.draw_me(screen)
            pygame.display.flip()
        
        # Timeout: if snake wanders too long without food
        if stepsSinceLastFood > 100:
            if PRINT_DEATH_REASON:
                print('timeout')
            break
        
        # Check border collision
        head_x, head_y = snake.segments[0]
        if not (0 <= head_x < GRID_DIMENSION and 0 <= head_y < GRID_DIMENSION):
            if PRINT_DEATH_REASON:
                print('out map')
            break
        
        # Check obstacle collision
        if snake.segments[0] in obstacles.positions:
            if PRINT_DEATH_REASON:
                print('hit obstacle')
            break
        
        # Snake control: human or AI
        snake_controller(snake, candy)
        
        # Check self-collision
        if snake.segments[0] in snake.segments[1:]:
            if PRINT_DEATH_REASON:
                print('hit snake')
            break
        
        # Candy eaten check with dynamic food and time efficiency bonus
        if snake.segments[0] == candy.location:
            time_bonus = max(0, (100 - stepsSinceLastFood) / 100)
            total_time_bonus += time_bonus
            stepsSinceLastFood = 0
            points += candy.value
            candy.relocate(candy_controller)
            # Ensure candy doesn't spawn on snake or obstacles
            while candy.location in snake.segments or candy.location in obstacles.positions:
                candy.relocate(candy_controller)
            # Grow snake (simple growth logic)
            snake.segments.append(snake.segments[0])
        
        # Move snake
        snake.move_me()
        
        if LIMIT_FPS:
            clock.tick(GAME_FPS)
        
        stepsSinceLastFood += 1
        steps_survived += 1
        # Dynamic food: increase value after a set time
        candy.time_on_board += 1
        if candy.time_on_board > 50:
            candy.value = 2
    
    # Survival bonus: capped to avoid survival-only strategies
    survival_bonus = min(steps_survived * 0.01, 10)
    # Final score includes time bonus and survival bonus, with a penalty for revisiting positions
    final_score = points + (total_time_bonus * 0.5) + survival_bonus - (snake.revisit_count * 0.05)
    return final_score

# ------------------------------------------------
# NEAT training functions
# ------------------------------------------------
def evaluate_generation(genomes, neat_config):
    for genome_id, genome_data in genomes:
        genome_data.fitness = 0
        brain = neat.nn.FeedForwardNetwork.create(genome_data, neat_config)
        ai_mover = ai_controller_builder(brain)
        for _ in range(ROUNDS_PER_AGENT):
            genome_data.fitness += run_game(ai_mover, random_location) / ROUNDS_PER_AGENT

def run_training(config_path, visualize_training=False):
    """
    Train the NEAT-based snake using the specified config file.
    The NEAT config file should be adjusted as follows (neat_config_modifications):
      - weight_mutate_power = 0.7  (was 0.5)
      - activation_default = sigmoid  (was tanh)
      - activation_options = sigmoid relu tanh
      - node_add_prob = 0.3  (was 0.2)
      - compatibility_threshold = 2.0  (was 3.0)
      - species_elitism = 2  (was 0)
      - Also, adjust the input neuron count from 28 to 52 to account for the expanded inputs.
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
    winner_genome = population.run(evaluate_generation, 1000)
    pickle.dump(winner_genome, open('best_genome.pkl', 'wb'))
    print('\nBest genome:\n{!s}'.format(winner_genome))

# ------------------------------------------------
# Human play and visualization functions
# ------------------------------------------------
def human_play():
    """Let a human control the snake using WASD keys."""
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

def visualize_bot(genome_path, config_path, candy_positions_path=None, obstacle_positions=None):
    """
    Watch a trained genome control the snake.
    Optionally use preset food positions and obstacles.
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
    final_score = run_game(snake_controller, candy_controller, obstacle_positions)
    print('Score:', final_score)

# ------------------------------------------------
# Entry point with argument parsing
# ------------------------------------------------
def entry_point():
    parser = argparse.ArgumentParser(description="Snake Game with NEAT Bot (Reworked with Modifications)")
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')
    train_sub = subparsers.add_parser('train', help='Train the bot using NEAT')
    train_sub.add_argument('--config', required=True, help='Path to NEAT config file')
    train_sub.add_argument('--watch', action='store_true', help='Watch the training process')
    play_sub = subparsers.add_parser('play', help='Play Snake as a human')
    watch_sub = subparsers.add_parser('watch', help='Watch the bot play Snake')
    watch_sub.add_argument('--genome', required=True, help='Path to genome file')
    watch_sub.add_argument('--config', required=True, help='Path to NEAT config file')
    watch_sub.add_argument('--food-pos', help='Path to food positions file (optional)')
    watch_sub.add_argument('--obstacles', help='Path to obstacles positions file (optional, pickled list of positions)')
    args = parser.parse_args()
    if args.mode == 'train':
        run_training(args.config, args.watch)
    elif args.mode == 'play':
        human_play()
    elif args.mode == 'watch':
        obstacle_positions = None
        if args.obstacles:
            obstacle_positions = pickle.load(open(args.obstacles, 'rb'))
        visualize_bot(args.genome, args.config, args.food_pos, obstacle_positions)
    else:
        parser.print_help()

if __name__ == '__main__':
    entry_point()
