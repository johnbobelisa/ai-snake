import pygame
import numpy as np
import pickle
import neat
from pygame.locals import (
    K_UP, K_DOWN, K_LEFT, K_RIGHT, K_w, K_a, K_s, K_d, K_1, K_2, K_3, K_4,
    KEYDOWN, QUIT, K_ESCAPE
)

# =============================================================================
#                             GAME CONSTANTS
# =============================================================================

BLOCK_SIZE = 25
GRID_DIMENSION = 16
SCREEN_SIZE = BLOCK_SIZE * GRID_DIMENSION

# Observation/Display settings
FRAMERATE = 10
USE_FRAMERATE = True
WATCH = True

# =============================================================================
#                          GAME OBJECTS & LOGIC
# =============================================================================

class Food:
    def __init__(self, position):
        self.position = tuple(position)

    def render(self, surface):
        """Draw the food with an inner border."""
        border = 5
        pos_px = np.multiply(self.position, BLOCK_SIZE)
        pygame.draw.rect(surface, (150, 0, 0), (*pos_px, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(surface, (255, 0, 0),
                         (pos_px[0] + border, pos_px[1] + border,
                          BLOCK_SIZE - 2 * border, BLOCK_SIZE - 2 * border))

    def respawn(self, position_func):
        """Respawn food at a new position."""
        self.position = tuple(position_func())

class Snake:
    def __init__(self, position):
        self.direction = self.move_right  # Default direction
        self.segments = [tuple(position)]  # List of segments starting with the head

    def render(self, surface):
        # Define colors for the gradient
        color_head = (255, 105, 180)  # Hot pink for the head
        color_tail = (0, 255, 255)    # Cyan for the tail
        N = len(self.segments)        # Number of segments

        # Draw each segment with a gradient color
        for i, segment in enumerate(self.segments):
            # Calculate interpolation factor (t) based on position
            if N > 1:
                t = i / (N - 1)  # From 0 (head) to 1 (tail)
            else:
                t = 0  # If only one segment, use head color

            # Interpolate between head and tail colors
            color = (
                int((1 - t) * color_head[0] + t * color_tail[0]),  # Red component
                int((1 - t) * color_head[1] + t * color_tail[1]),  # Green component
                int((1 - t) * color_head[2] + t * color_tail[2])   # Blue component
            )

            # Darker border color (70% of the segment color)
            border_color = (
                max(0, int(color[0] * 0.7)),
                max(0, int(color[1] * 0.7)),
                max(0, int(color[2] * 0.7))
            )

            # Convert segment grid position to pixel position
            seg_px = np.multiply(segment, BLOCK_SIZE)

            # Draw outer rectangle (border)
            pygame.draw.rect(surface, border_color, (*seg_px, BLOCK_SIZE, BLOCK_SIZE))

            # Draw inner rectangle (main body)
            border = 5
            pygame.draw.rect(surface, color,
                             (seg_px[0] + border, seg_px[1] + border,
                              BLOCK_SIZE - 2 * border, BLOCK_SIZE - 2 * border))

        # Draw eyes on the head segment
        if N > 0:
            head = self.segments[0]
            head_px = np.multiply(head, BLOCK_SIZE)
            eye_radius = 3    # Size of the white part of the eye
            pupil_radius = 1  # Size of the black pupil

            # Eye positions relative to the head’s top-left corner, based on direction
            eye_offsets = {
                self.move_right: [(18, 7), (18, 18)],
                self.move_left: [(7, 7), (7, 18)],
                self.move_up: [(7, 7), (18, 7)],
                self.move_down: [(7, 18), (18, 18)]
            }

            # Pupil offsets to make eyes look in the movement direction
            pupil_offset = {
                self.move_right: (2, 0),   # Right
                self.move_left: (-2, 0),   # Left
                self.move_up: (0, -2),     # Up
                self.move_down: (0, 2)     # Down
            }

            # Draw eyes if the direction is recognized
            if self.direction in eye_offsets:
                for offset in eye_offsets[self.direction]:
                    eye_pos = (head_px[0] + offset[0], head_px[1] + offset[1])
                    # White circle for the eye
                    pygame.draw.circle(surface, (255, 255, 255), eye_pos, eye_radius)
                    # Black pupil offset in the direction of movement
                    pupil_pos = (eye_pos[0] + pupil_offset[self.direction][0],
                                 eye_pos[1] + pupil_offset[self.direction][1])
                    pygame.draw.circle(surface, (0, 0, 0), pupil_pos, pupil_radius)

    def move(self):
        """Advance the snake in its current direction."""
        new_head = self.direction(self.segments[0])
        self.segments.insert(0, new_head)
        self.segments.pop()

    # Directional moves
    def move_left(self, head):
        return (head[0] - 1, head[1])

    def move_right(self, head):
        return (head[0] + 1, head[1])

    def move_up(self, head):
        return (head[0], head[1] - 1)

    def move_down(self, head):
        return (head[0], head[1] + 1)


def get_random_position():
    """Return a random (x, y) position within the grid."""
    return np.random.randint(0, GRID_DIMENSION, size=2)


def handle_human_input(snake, food):
    """Adjust snake direction based on user key presses."""
    keys = pygame.key.get_pressed()
    
    # Prevent 180-degree turns (moving directly into yourself)
    current_direction = snake.direction
    
    if (keys[K_UP] or keys[K_w]) and current_direction != snake.move_down:
        snake.direction = snake.move_up
    elif (keys[K_DOWN] or keys[K_s]) and current_direction != snake.move_up:
        snake.direction = snake.move_down
    elif (keys[K_LEFT] or keys[K_a]) and current_direction != snake.move_right:
        snake.direction = snake.move_left
    elif (keys[K_RIGHT] or keys[K_d]) and current_direction != snake.move_left:
        snake.direction = snake.move_right


def play_game(snake_control, food_control, auto_restart=False, watch=True):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Snake Game")

    # Try to load background or create a simple one
    try:
        background = pygame.image.load("bg.png").convert()
        background = pygame.transform.scale(background, (SCREEN_SIZE, SCREEN_SIZE))
    except:
        background = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE))
        background.fill((0, 0, 0))
    
    # Initialize game objects
    snake = Snake(get_random_position())
    food = Food(get_random_position())
    
    score = 0
    search_duration = 0
    running = True

    while running:
        # Process events (exit on ESC or window close)
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False
            elif event.type == QUIT:
                running = False

        # Render only if watch is True
        if watch:
            screen.blit(background, (0, 0))
            snake.render(screen)
            food.render(screen)
            # Display score
            font = pygame.font.SysFont(None, 24)
            score_text = font.render(f"Score: {score}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))
            pygame.display.flip()

        # Check for timeout
        if search_duration > 100:
            break

        # End game if snake head is out of bounds
        head_x, head_y = snake.segments[0]
        if not (0 <= head_x < GRID_DIMENSION and 0 <= head_y < GRID_DIMENSION):
            break

        # Update snake direction
        snake_control(snake, food)

        # End game if snake collides with itself
        if snake.segments[0] in snake.segments[1:]:
            break

        # Check if food is eaten
        if snake.segments[0] == food.position:
            search_duration = 0
            score += 1
            food.respawn(food_control)
            while food.position in snake.segments:
                food.respawn(food_control)
            snake.segments.append(snake.segments[-1])
        
        # Advance the snake
        snake.move()
        if USE_FRAMERATE:
            clock.tick(FRAMERATE)
        search_duration += 1

    # Handle game over based on auto_restart
    if not auto_restart and watch:  # Only show game over if watching and not auto-restarting
        font = pygame.font.SysFont(None, 48)
        game_over = font.render("Game Over", True, (255, 0, 0))
        final_score = font.render(f"Final Score: {score}", True, (255, 255, 255))
        restart_text = font.render("Press any key to restart", True, (255, 255, 255))
        
        screen.blit(game_over, (SCREEN_SIZE//2 - game_over.get_width()//2, SCREEN_SIZE//2 - 60))
        screen.blit(final_score, (SCREEN_SIZE//2 - final_score.get_width()//2, SCREEN_SIZE//2))
        screen.blit(restart_text, (SCREEN_SIZE//2 - restart_text.get_width()//2, SCREEN_SIZE//2 + 60))
        pygame.display.flip()
        
        # Wait for key press to restart or quit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    waiting = False
                    pygame.quit()
                    return score
                elif event.type == KEYDOWN:
                    waiting = False
    
    pygame.quit()  # Close the display after each game
    return score


# =============================================================================
#                     BOT / NEAT TRAINING FUNCTIONS
# =============================================================================

PLAYS_PER_BOT = 3
VISION_RANGE = 5


def create_bot_controller(model):
    """
    Returns a function that directs the snake using the given neural network.
    """
    def bot_control(snake, food):
        state = extract_local_state(snake, food)
        output = model.activate(state)
        decision = np.argmax(output)
        if decision == 0:
            snake.direction = snake.move_up
        elif decision == 1:
            snake.direction = snake.move_right
        elif decision == 2:
            snake.direction = snake.move_down
        elif decision == 3:
            snake.direction = snake.move_left
    return bot_control


def is_position_occupied(position, snake):
    """Return 1 if the given grid cell is occupied or out of bounds; otherwise, 0."""
    if position in snake.segments:
        return 1
    if not (0 <= position[0] < GRID_DIMENSION and 0 <= position[1] < GRID_DIMENSION):
        return 1
    return 0


def extract_local_state(snake, food):
    """
    Returns a binary state vector representing the grid around the snake's head,
    plus booleans for food direction.
    """
    state = []
    head = snake.segments[0]
    
    # Create a VISION_RANGE x VISION_RANGE window (excluding head cell)
    for i in range(-VISION_RANGE // 2 + 1, VISION_RANGE // 2 + 1):
        for j in range(-VISION_RANGE // 2 + 1, VISION_RANGE // 2 + 1):
            if i != 0 or j != 0:
                state.append(is_position_occupied((head[0] + j, head[1] + i), snake))
    
    # Append booleans indicating food location relative to the head
    food_direction = (
        1 if food.position[1] < head[1] else 0,
        1 if food.position[1] > head[1] else 0,
        1 if food.position[0] < head[0] else 0,
        1 if food.position[0] > head[0] else 0
    )
    state.extend(food_direction)
    return state


def evaluate_generation(genomes, config):
    """
    Calculate the fitness for each genome.
    """
    for genome_id, genome in genomes:
        genome.fitness = 0
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        bot_control = create_bot_controller(network)
        for _ in range(PLAYS_PER_BOT):
            genome.fitness += play_game(bot_control, get_random_position, auto_restart=True, watch=False) / PLAYS_PER_BOT


def start_neat_training(config_path):
    """
    Train the neural network using NEAT.
    """
    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats_reporter = neat.StatisticsReporter()
    population.add_reporter(stats_reporter)
    population.add_reporter(neat.Checkpointer(50))
    
    global USE_FRAMERATE, FRAMERATE, WATCH
    USE_FRAMERATE = False  # Run as fast as possible
    FRAMERATE = 100  # Kept as a fallback, though not used
    WATCH = False  # Not necessary, but consistent with watch=False in play_game
    
    best_genome = population.run(evaluate_generation, 50)
    pickle.dump(best_genome, open('best_genome.pkl', 'wb'))
    pickle.dump(config, open('best_config.pkl', 'wb'))
    print("\nBest genome saved as 'best_genome.pkl'")


def watch_ai_play(genome_path, config_path):
    """
    Continuously play games using the saved best genome.
    """
    try:
        best_genome = pickle.load(open(genome_path, 'rb'))
        config = pickle.load(open(config_path, 'rb'))
        network = neat.nn.FeedForwardNetwork.create(best_genome, config)
        
        global WATCH, USE_FRAMERATE, FRAMERATE
        WATCH = True
        USE_FRAMERATE = True
        FRAMERATE = 10
        
        bot_control = create_bot_controller(network)
        
        while True:
            score = play_game(bot_control, get_random_position, auto_restart=True, watch=True)
            print(f"AI Score: {score}")
    except Exception as e:
        print(f"Error loading AI: {e}")
        print("Make sure you've trained the AI first or have valid genome/config files")
        return


def play_human_game():
    """
    Start a game controlled by a human player.
    """
    global FRAMERATE
    FRAMERATE = 10
    while True:
        score = play_game(handle_human_input, get_random_position, auto_restart=False, watch=True)
        print(f"Human Score: {score}")


# =============================================================================
#                                MAIN MENU
# =============================================================================

def main_menu():
    """Display a menu to select game mode."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Snake Game - Main Menu")
    
    # Create a simple background
    background = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE))
    background.fill((0, 0, 20))
    
    font_large = pygame.font.SysFont(None, 48)
    font_medium = pygame.font.SysFont(None, 36)
    
    title = font_large.render("SNAKE GAME", True, (0, 255, 0))
    option1 = font_medium.render("1. Play as Human", True, (255, 255, 255))
    option2 = font_medium.render("2. Watch AI Play", True, (255, 255, 255))
    option3 = font_medium.render("3. Train New AI", True, (255, 255, 255))
    option4 = font_medium.render("4. Quit", True, (255, 255, 255))
    
    running = True
    while running:
        screen.blit(background, (0, 0))
        
        # Center the title and options
        screen.blit(title, (SCREEN_SIZE//2 - title.get_width()//2, 100))
        screen.blit(option1, (SCREEN_SIZE//2 - option1.get_width()//2, 200))
        screen.blit(option2, (SCREEN_SIZE//2 - option2.get_width()//2, 250))
        screen.blit(option3, (SCREEN_SIZE//2 - option3.get_width()//2, 300))
        screen.blit(option4, (SCREEN_SIZE//2 - option4.get_width()//2, 350))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                pygame.quit()
                return
            
            if event.type == KEYDOWN:
                if event.key == K_1 or event.key == pygame.K_KP1:
                    pygame.quit()
                    play_human_game()
                    return
                elif event.key == K_2 or event.key == pygame.K_KP2:
                    pygame.quit()
                    watch_ai_play('best_genome.pkl', 'best_config.pkl')
                    return
                elif event.key == K_3 or event.key == pygame.K_KP3:
                    pygame.quit()
                    try:
                        start_neat_training('config-feedforward.txt')
                    except Exception as e:
                        print(f"Error starting training: {e}")
                        print("Make sure config-feedforward.txt is available")
                    return
                elif event.key == K_4 or event.key == pygame.K_KP4 or event.key == K_ESCAPE:
                    running = False
                    pygame.quit()
                    return


# =============================================================================
#                                MAIN
# =============================================================================

if __name__ == '__main__':
    main_menu()