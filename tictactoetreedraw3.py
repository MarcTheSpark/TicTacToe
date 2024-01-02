# Pointing out that the branches are deceptive, because they don't represent the same board state
# FOR EACH MOVE, MAKE Y Axis every possible board state after those moves.
# That way we can see forking and branching paths
import numpy as np
from boardstates import all_board_states
from tictactoe import extrapolate_all_games, remove_near_duplicates, start_game

def get_board_state_index(board_state, move_num):
    if not isinstance(board_state, np.ndarray):
        board_state = np.array(board_state)
    this_move_states = all_board_states[move_num]
    
    matches = np.where(np.all(this_move_states == board_state, axis=1))[0]
    if len(matches) > 0:
        return matches[0]
    else:
        return None

games = extrapolate_all_games([start_game], skill=2)
games = [game.reshape((-1, 9)) for game in games]
# games = remove_near_duplicates(games)
print(games[20])

# ---------------------------- DRAWING -------------------------------

import pygame

SCREEN_DIM = 1920, 1080
MARGINS = 100, 100

x_step = (SCREEN_DIM[0] - 2 * MARGINS[0]) / 8
y_step = (SCREEN_DIM[1] - 2 * MARGINS[1]) / 8

def draw_game(screen, move_dict, origin=None):
    if origin:
        for x in move_dict:
            endpoint = origin[0] + x_step, MARGINS[1] + (x - 1) * y_step
            pygame.draw.line(screen, (255, 255, 255), origin, endpoint, 2)
            draw_move_lines(screen, move_dict[x], endpoint)
    else:
        for x in move_dict:
            draw_move_lines(screen, move_dict[x], (MARGINS[0], MARGINS[1] + (x - 1) * y_step))

            
def main():
    # Initialize pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode(SCREEN_DIM)

    # Set running to True to keep the window open
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the screen with a black color
        screen.fill((0, 0, 0))

        # Draw a single white line segment
        draw_move_lines(screen, all_moves)
                
        # Update the display
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()