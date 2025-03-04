"""
Code for running the snake game using pygame

Author: John Elsa
Date: 04/03/2025 (dd/mm/yyyy)
"""

import pygame
import random
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    KEYDOWN,
    QUIT,
    K_ESCAPE
)

# Board settings
BLOCK_SIZE = 20
GRID_SIZE = 20
SCREEN_SIZE = BLOCK_SIZE * GRID_SIZE

# Game settings
FRAME_RATE = 20            # Target FPS for rendering
MOVE_DELAY = 150           # Milliseconds between snake moves
USE_FRAMERATE = True       # For consistency with frame rate control
WATCH = True               # (Reserved for future debugging or visualization)
SHOW_DEATH_CAUSE = True    # Display cause of death on game over

class Food:
    """Class representing the food the snake eats."""
    def __init__(self, snake_body):
        self.position = self.randomize(snake_body)
    
    def randomize(self, snake_body):
        """Randomly place food on the grid avoiding the snake's body."""
        while True:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if pos not in snake_body:
                return pos

    def draw(self, surface):
        """Draw the food as a red block."""
        rect = pygame.Rect(self.position[0] * BLOCK_SIZE, self.position[1] * BLOCK_SIZE,
                           BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(surface, (255, 0, 0), rect)

class Snake:
    """Class representing the snake."""
    def __init__(self):
        # Start at the center of the grid, moving to the right.
        self.body = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = (1, 0)  # (dx, dy): moving right
        self.grow_flag = False

    def change_direction(self, new_direction):
        """Change snake direction if not directly opposite to current direction."""
        # Prevent reversing direction (e.g., if moving right, ignore left)
        if (new_direction[0] * -1, new_direction[1] * -1) == self.direction:
            return
        self.direction = new_direction

    def move(self):
        """Move the snake in the current direction."""
        head = self.body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.body.insert(0, new_head)
        if not self.grow_flag:
            self.body.pop()  # Remove tail if not growing
        else:
            self.grow_flag = False

    def grow(self):
        """Set flag to grow snake (by not removing tail on next move)."""
        self.grow_flag = True

    def check_collision(self):
        """
        Check for collisions:
          - with walls
          - with itself
        Returns a tuple (collision: bool, cause: str).
        """
        head = self.body[0]
        # Check wall collisions
        if head[0] < 0 or head[0] >= GRID_SIZE or head[1] < 0 or head[1] >= GRID_SIZE:
            return True, "Hit wall"
        # Check self collision
        if head in self.body[1:]:
            return True, "Hit itself"
        return False, ""

    def draw(self, surface):
        """Draw the snake as green blocks."""
        for segment in self.body:
            rect = pygame.Rect(segment[0] * BLOCK_SIZE, segment[1] * BLOCK_SIZE,
                               BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(surface, (0, 255, 0), rect)

class Game:
    """Class to manage the game state, updating logic, and drawing."""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        # Pre-create fonts for score and game-over messages
        self.font_score = pygame.font.SysFont("Arial", 24)
        self.font_game_over = pygame.font.SysFont("Arial", 36)
        self.reset()

    def reset(self):
        """Reset the game state to start a new game."""
        self.snake = Snake()
        self.food = Food(self.snake.body)
        self.score = 0
        self.game_over = False
        self.death_cause = ""

    def process_events(self):
        """Process user input events."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.game_over = True
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.game_over = True
                elif event.key == K_UP:
                    self.snake.change_direction((0, -1))
                elif event.key == K_DOWN:
                    self.snake.change_direction((0, 1))
                elif event.key == K_LEFT:
                    self.snake.change_direction((-1, 0))
                elif event.key == K_RIGHT:
                    self.snake.change_direction((1, 0))

    def update(self):
        """Update the game state: move the snake, check for food and collisions."""
        self.snake.move()
        # Check if the snake eats the food.
        if self.snake.body[0] == self.food.position:
            self.snake.grow()
            self.score += 1
            self.food = Food(self.snake.body)
        # Check for collisions.
        collision, cause = self.snake.check_collision()
        if collision:
            self.game_over = True
            if SHOW_DEATH_CAUSE:
                self.death_cause = cause

    def draw(self):
        """Draw the game state to the screen."""
        self.screen.fill((0, 0, 0))
        self.snake.draw(self.screen)
        self.food.draw(self.screen)
        # Draw the score using the pre-created font.
        score_surface = self.font_score.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_surface, (5, 5))
        pygame.display.flip()

    def show_game_over(self):
        """Display game over message and then quit."""
        msg = "Game Over"
        if SHOW_DEATH_CAUSE and self.death_cause:
            msg += f": {self.death_cause}"
        msg_surface = self.font_game_over.render(msg, True, (255, 255, 255))
        # Center the message on screen.
        rect = msg_surface.get_rect(center=(SCREEN_SIZE // 2, SCREEN_SIZE // 2))
        self.screen.blit(msg_surface, rect)
        pygame.display.flip()
        pygame.time.wait(2000)
        pygame.quit()

    def play(self):
        """Main loop for playing the game manually using a fixed timestep."""
        accumulator = 0  # Time accumulator for fixed update rate (in milliseconds)
        while not self.game_over:
            dt = self.clock.tick(FRAME_RATE)  # dt in milliseconds; caps FPS at FRAME_RATE
            accumulator += dt
            self.process_events()
            # Update snake movement at fixed MOVE_DELAY intervals.
            while accumulator >= MOVE_DELAY:
                self.update()
                accumulator -= MOVE_DELAY
            self.draw()
        self.show_game_over()

def play():
    """Top-level function to run the game."""
    game = Game()
    game.play()

if __name__ == '__main__':
    play()
