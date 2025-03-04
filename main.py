# Import necessary libraries for game, NEAT training, and argument parsing
import pygame
import neat
import numpy as np
import pickle
import argparse
from pygame.locals import (
    K_w, K_a, K_s, K_d,  # Keys for human control (WASD)
    KEYDOWN, QUIT, K_ESCAPE  # Event types and escape key
)

# Define game board constants
BLOCK_SIZE = 30  # Size of each grid block in pixels
GRID_SIZE = 30   # Number of blocks per side of the grid
SCREEN_SIZE = BLOCK_SIZE * GRID_SIZE  # Total screen size in pixels

# Default rendering settings (will be adjusted based on mode)
WATCH = False         # Whether to display the game visually
USE_FRAMERATE = False # Whether to limit the frame rate
FRAMERATE = 10        # Frames per second (default value)
SHOW_DEATH_CAUSE = False  # Whether to print why the snake died

# NEAT-specific settings
PLAYS_PER_BOT = 3  # Number of games to average for each bot's fitness
VISION_BOX = 5     # Size of the vision grid around the snake's head

# Define the Food class to manage food placement and drawing
class Food:
    def __init__(self, pos):
        self.pos = tuple(pos)  # Store food position as a tuple

    def draw(self, screen):
        """Draw the food on the screen with a border"""
        border = 5  # Border size in pixels
        real_pos = np.multiply(self.pos, BLOCK_SIZE)  # Convert grid pos to pixels
        # Draw outer rectangle (darker red)
        pygame.draw.rect(screen, (150, 0, 0), (*real_pos, BLOCK_SIZE, BLOCK_SIZE))
        # Draw inner rectangle (brighter red)
        pygame.draw.rect(screen, (255, 0, 0),
                         (real_pos[0] + border, real_pos[1] + border,
                          BLOCK_SIZE - 2 * border, BLOCK_SIZE - 2 * border))

    def respawn(self, func):
        """Respawn food at a new position using the provided function"""
        new_pos = tuple(func())  # Get new position from food_controller
        self.pos = new_pos

# Define the Snake class to manage snake movement and drawing
class Snake:
    def __init__(self, pos):
        self.dir = self.right  # Default direction is right
        self.blocks = [tuple(pos)]  # List of block positions, starting with head

    def draw(self, screen):
        """Draw each snake block on the screen with a border"""
        border = 5  # Border size in pixels
        for pos in self.blocks:
            real_pos = np.multiply(pos, BLOCK_SIZE)  # Convert grid pos to pixels
            # Draw outer rectangle (darker green)
            pygame.draw.rect(screen, (0, 150, 0), (*real_pos, BLOCK_SIZE, BLOCK_SIZE))
            # Draw inner rectangle (brighter green)
            pygame.draw.rect(screen, (0, 255, 0),
                             (real_pos[0] + border, real_pos[1] + border,
                              BLOCK_SIZE - 2 * border, BLOCK_SIZE - 2 * border))

    def move(self):
        """Move the snake in the current direction"""
        new_head = self.dir(self.blocks[0])  # Calculate new head position
        self.blocks.insert(0, new_head)      # Add new head
        self.blocks.pop()                    # Remove tail

    # Direction functions to compute new head position
    def left(self, head):
        return (head[0] - 1, head[1])
    def right(self, head):
        return (head[0] + 1, head[1])
    def up(self, head):
        return (head[0], head[1] - 1)
    def down(self, head):
        return (head[0], head[1] + 1)

def rand_pos():
    """Return a random position within the grid"""
    return np.random.randint(0, GRID_SIZE, size=2)

def human_mover(snake, food):
    """Control snake direction based on human key presses (WASD)"""
    presses = pygame.key.get_pressed()
    if presses[K_w]:
        snake.dir = snake.up
    elif presses[K_s]:
        snake.dir = snake.down
    elif presses[K_a]:
        snake.dir = snake.left
    elif presses[K_d]:
        snake.dir = snake.right

def play(snake_controller, food_controller):
    """Play a game of Snake with given controllers and return the score"""
    global WATCH, USE_FRAMERATE, FRAMERATE, SHOW_DEATH_CAUSE

    # Initialize PyGame and set up the display
    pygame.init()
    screen = pygame.display.set_mode([SCREEN_SIZE, SCREEN_SIZE])
    clock = pygame.time.Clock()

    # Create initial snake and food objects
    snake = Snake(rand_pos())
    food = Food(food_controller())  # Use food_controller to get initial pos

    score = 0         # Number of foods eaten
    search_length = 0 # Steps since last food (to detect timeouts)

    done = False
    while not done:
        # Handle quit events
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                done = True
            elif event.type == QUIT:
                done = True

        # Render the game if watching is enabled
        if WATCH:
            screen.fill((0, 0, 0))  # Clear screen (black background)
            snake.draw(screen)      # Draw snake
            food.draw(screen)       # Draw food
            pygame.display.flip()   # Update display

        # Check for timeout (snake takes too long to find food)
        if search_length > 100:
            if SHOW_DEATH_CAUSE:
                print('timeout')
            break

        # Check if snake hits the border
        head_x, head_y = snake.blocks[0]
        if not (0 <= head_x < GRID_SIZE and 0 <= head_y < GRID_SIZE):
            if SHOW_DEATH_CAUSE:
                print('out map')
            break

        # Update snake direction using the controller
        snake_controller(snake, food)

        # Check if snake hits itself
        if snake.blocks[0] in snake.blocks[1:]:
            if SHOW_DEATH_CAUSE:
                print('hit snake')
            break

        # Check if snake eats food
        if snake.blocks[0] == food.pos:
            search_length = 0           # Reset timeout counter
            score += 1                  # Increment score
            food.respawn(food_controller)  # Respawn food
            # Ensure food doesn't spawn on snake
            while food.pos in snake.blocks:
                food.respawn(food_controller)
            snake.blocks.append(snake.blocks[0])  # Grow snake

        snake.move()  # Move snake forward

        # Apply frame rate limit if enabled
        if USE_FRAMERATE:
            clock.tick(FRAMERATE)

        search_length += 1  # Increment steps since last food

    return score

def bot_mover_maker(model):
    """Create a bot mover function using a NEAT neural network model"""
    def bot_mover(snake, food):
        state = local_state(snake, food)  # Get current game state
        guesses = model.activate(state)   # Get model predictions
        new_dir = np.argmax(guesses)      # Choose direction with highest score
        # Set snake direction based on model output
        if new_dir == 0:
            snake.dir = snake.up
        elif new_dir == 1:
            snake.dir = snake.right
        elif new_dir == 2:
            snake.dir = snake.down
        elif new_dir == 3:
            snake.dir = snake.left
    return bot_mover

def is_occupied(pos, snake):
    """Check if a position is occupied (by snake or border)"""
    if pos in snake.blocks:  # Position is part of snake
        return 1
    # Position is outside grid boundaries
    if not (0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE):
        return 1
    return 0  # Position is free

def local_state(snake, food):
    """Compute the snake's local state for the NEAT model"""
    state = []
    head = snake.blocks[0]  # Snake's head position

    # Build a VISION_BOX x VISION_BOX grid around the head (excluding head itself)
    for i in range(-VISION_BOX // 2 + 1, VISION_BOX // 2 + 1):
        for j in range(-VISION_BOX // 2 + 1, VISION_BOX // 2 + 1):
            if i != 0 or j != 0:  # Skip the head position
                state.append(is_occupied((head[0] + j, head[1] + i), snake))

    # Add food direction indicators (up, down, left, right)
    food_direction = (
        1 if food.pos[1] < head[1] else 0,  # Food is above
        1 if food.pos[1] > head[1] else 0,  # Food is below
        1 if food.pos[0] < head[0] else 0,  # Food is left
        1 if food.pos[0] > head[0] else 0   # Food is right
    )
    state += list(food_direction)
    return state

def train_generation(genomes, config):
    """Evaluate fitness of genomes in a NEAT generation"""
    for _, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0  # Initialize fitness
        model = neat.nn.FeedForwardNetwork.create(genome, config)  # Create NN
        bot_mover = bot_mover_maker(model)  # Get bot controller
        # Average fitness over multiple plays
        for _ in range(PLAYS_PER_BOT):
            genome.fitness += play(bot_mover, rand_pos) / PLAYS_PER_BOT

def train(config_file, watch_training=False):
    """Train a Snake bot using the NEAT algorithm"""
    global WATCH, USE_FRAMERATE, FRAMERATE

    # Set rendering settings based on whether we're watching
    if watch_training:
        WATCH = True
        USE_FRAMERATE = True
        FRAMERATE = 100  # Faster frame rate for watching training
    else:
        WATCH = False    # No rendering to speed up training
        USE_FRAMERATE = False  # No frame rate limit

    # Load NEAT configuration from file
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create NEAT population
    p = neat.Population(config)

    # Add reporters for progress tracking
    p.add_reporter(neat.StdOutReporter(True))  # Print progress to console
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))  # Save checkpoints every 50 gens

    # Run NEAT for up to 1000 generations
    winner = p.run(train_generation, 1000)

    # Save the best genome to a file
    pickle.dump(winner, open('best_genome.pkl', 'wb'))

    # Display the best genome
    print('\nBest genome:\n{!s}'.format(winner))

def play_human():
    """Play Snake as a human using WASD controls"""
    global WATCH, USE_FRAMERATE, FRAMERATE

    # Set rendering settings for human play
    WATCH = True           # Must watch to play
    USE_FRAMERATE = True   # Limit frame rate
    FRAMERATE = 10         # Suitable speed for human control

    # Play one game and print the score
    score = play(human_mover, rand_pos)
    print(f"Score: {score}")

def preset_food_pos_maker(positions):
    """Create a food controller that uses preset positions"""
    pos = positions[:]  # Copy list to avoid modifying original
    def preset_food_pos():
        try:
            return pos.pop(0)  # Return next position
        except IndexError:
            print('out of given positions; using random ones')
            return rand_pos()  # Fall back to random if out of positions
    return preset_food_pos

def watch_bot(genome_file, config_file, food_pos_file=None):
    """Watch a trained bot play Snake"""
    global WATCH, USE_FRAMERATE, FRAMERATE

    # Set rendering settings for watching
    WATCH = True
    USE_FRAMERATE = True
    FRAMERATE = 30  # Suitable speed for observation

    # Load the trained genome
    genome = pickle.load(open(genome_file, 'rb'))

    # Load NEAT config (assuming text config file like in training)
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create neural network model from genome
    model = neat.nn.FeedForwardNetwork.create(genome, config)
    snake_controller = bot_mover_maker(model)  # Get bot controller

    # Set food controller based on whether positions are provided
    if food_pos_file:
        food_positions = pickle.load(open(food_pos_file, 'rb'))
        food_controller = preset_food_pos_maker(food_positions)
    else:
        food_controller = rand_pos  # Use random positions

    # Play the game and print the score
    score = play(snake_controller, food_controller)
    print('Score:', score)

def main():
    """Parse command-line arguments and run the selected mode"""
    parser = argparse.ArgumentParser(description="Snake Game with NEAT Bot")
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')

    # Train mode parser
    train_parser = subparsers.add_parser('train', help='Train the bot using NEAT')
    train_parser.add_argument('--config', required=True, help='Path to NEAT config file')
    train_parser.add_argument('--watch', action='store_true', help='Watch the training process')

    # Play mode parser
    play_parser = subparsers.add_parser('play', help='Play Snake as a human')

    # Watch mode parser
    watch_parser = subparsers.add_parser('watch', help='Watch the bot play Snake')
    watch_parser.add_argument('--genome', required=True, help='Path to genome file')
    watch_parser.add_argument('--config', required=True, help='Path to NEAT config file')
    watch_parser.add_argument('--food-pos', help='Path to food positions file (optional)')

    # Parse arguments
    args = parser.parse_args()

    # Execute the selected mode
    if args.mode == 'train':
        train(args.config, args.watch)
    elif args.mode == 'play':
        play_human()
    elif args.mode == 'watch':
        watch_bot(args.genome, args.config, args.food_pos)
    else:
        parser.print_help()  # Show help if no mode is specified

if __name__ == '__main__':
    main()