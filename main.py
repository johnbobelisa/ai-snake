import pygame
import random

#  ----------------------------------------- Constants -----------------------------------------
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
GRID_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
UP = (0, -1)
DOWN = (0, 1)
RIGHT = (1, 0)
LEFT = (-1, 0)

pygame.font.init()
game_font = pygame.font.SysFont(None, 36)
title_font = pygame.font.SysFont(None, 72)

#  ----------------------------------------- Constants -----------------------------------------

#  ----------------------------------------- Game States & Entities -----------------------------------------

# Snake initial position and properties (Start at center and look right)
# snake_positions[0] is head of snake, and the last tuple is the tail
snake_positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
snake_direction = RIGHT
eye_size = GRID_SIZE // 5
eye_offset = GRID_SIZE // 4

# Food
food_position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
food_exists = True

# Game states
running = True
game_over = False
score = 0

#  ----------------------------------------- Game States & Entities -----------------------------------------

#  ----------------------------------------- Game Loop -----------------------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()

while running:
    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Pygame.QUIT means user clicked X to close window 
            running = False
        
        # Handling keyboard events
        elif event.type == pygame.KEYDOWN:
            if not game_over:
                # snake_direction != ... is a check to ensure snake can only move 1 direction at a time
                if event.key == pygame.K_UP and snake_direction != DOWN:
                    snake_direction = UP
                elif event.key == pygame.K_DOWN and snake_direction != UP:
                    snake_direction = DOWN
                elif event.key == pygame.K_LEFT and snake_direction != RIGHT:
                    snake_direction = LEFT
                elif event.key == pygame.K_RIGHT and snake_direction != LEFT:
                    snake_direction = RIGHT

            # Restart game when R is pressed
            if game_over and event.key == pygame.K_r:
                snake_positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
                snake_direction = RIGHT
                food_position = (random.randint(0, GRID_WIDTH - 1), 
                                random.randint(0, GRID_HEIGHT - 1))
                food_exists = True
                game_over = False
                score = 0
        
    # Game Logic
    if not game_over:
        # Move the snake
        head_x, head_y = snake_positions[0]
        new_head = (head_x + snake_direction[0], head_y + snake_direction[1])

        # Check for collisions:
        # Wall collisions
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            game_over = True
        
        # Self collisions
        elif new_head in snake_positions[1:]:
            game_over = True

        # No collision -> Move the snake
        if not game_over:
            snake_positions.insert(0, new_head)
            
            # Check if snake ate food
            if snake_positions[0] == food_position:
                food_exists = False
                score += 1
            else:
                snake_positions.pop()
        

    # Fill the screen with black to wipe anything from last frame
    screen.fill(BLACK) 
    
    # Draw each segment of the snake
    for i, position in enumerate(snake_positions):
        if i == 0: # head
            # Draw Head
            pygame.draw.rect(screen, (0, 200, 0), pygame.Rect(position[0] * GRID_SIZE, position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
            # Left eye
            pygame.draw.rect(screen, (255, 255, 255),
                           pygame.Rect(position[0] * GRID_SIZE + eye_offset, 
                                     position[1] * GRID_SIZE + eye_offset,
                                     eye_size, eye_size))
            
            # Right eye
            pygame.draw.rect(screen, (255, 255, 255),
                           pygame.Rect(position[0] * GRID_SIZE + GRID_SIZE - eye_offset - eye_size, 
                                     position[1] * GRID_SIZE + eye_offset,
                                     eye_size, eye_size))
            
        else: # Body
            # Alternate colors for body segments
            color = (0, 255, 0) if i % 2 == 0 else (0, 220, 0)
            pygame.draw.rect(screen, color, pygame.Rect(position[0] * GRID_SIZE, position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    
    # Generate new food if needed
    if not food_exists:
        food_position = (random.randint(0, GRID_WIDTH - 1), 
                        random.randint(0, GRID_HEIGHT - 1))
        food_exists = True
    
    # Draw the food (apple-like)
    pygame.draw.rect(screen, RED,pygame.Rect(food_position[0] * GRID_SIZE, food_position[1] * GRID_SIZE,GRID_SIZE, GRID_SIZE))
    
    # Draw stem
    pygame.draw.rect(screen, (101, 67, 33), pygame.Rect(food_position[0] * GRID_SIZE + GRID_SIZE // 2 - 2, food_position[1] * GRID_SIZE - 5, 4, 5))
    
    # Display score and speed
    score_text = game_font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    # Speed increases with score
    current_speed = 10 + min(score // 5, 15)  # Max speed increase of 15
    speed_text = game_font.render(f"Speed: {current_speed}", True, (255, 255, 255))
    screen.blit(speed_text, (10, 50))

    # Display game over message
    if game_over:
        # Dim the screen a bit
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Semi-transparent black
        screen.blit(overlay, (0, 0))
        
        # Game over text
        game_over_text = title_font.render("GAME OVER", True, (255, 0, 0))
        screen.blit(game_over_text, 
                   (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 
                    SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2))
        
        # Final score
        final_score_text = game_font.render(f"Final Score: {score}", True, (255, 255, 255))
        screen.blit(final_score_text, 
                   (SCREEN_WIDTH // 2 - final_score_text.get_width() // 2, 
                    SCREEN_HEIGHT // 2 + 20))
        
        # Restart text
        restart_text = game_font.render("Press R to Restart", True, (255, 255, 255))
        screen.blit(restart_text, 
                   (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 
                    SCREEN_HEIGHT // 2 + 60))

    # Double buffer Rendering
    pygame.display.flip()
    # Game speed increases with score
    current_speed = 10 + min(score // 5, 15)  # Max speed increase of 15
    clock.tick(current_speed)

pygame.quit()

#  ----------------------------------------- Game Loop -----------------------------------------
