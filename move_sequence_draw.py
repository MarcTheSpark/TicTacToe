from game_extrapolation import extrapolate_all_games, remove_near_duplicates
from specific_games import start_game
from utils import flat_format
import numpy as np


games = extrapolate_all_games([start_game], prune_symmetrical=True, skill=4)
games = remove_near_duplicates(games)


flattened = [flat_format(x) for x in games]
move_sequences = [[int(np.argwhere(array[i] != array[i-1])[0]) + 1 for i in range(1, len(array))]
                  for array in flattened]


all_moves = {}
for move_sequence in move_sequences:
    current_move_dict = all_moves
    for move in move_sequence:
        if move not in current_move_dict:
            current_move_dict[move] = {}
        current_move_dict = current_move_dict[move]


# -------------------------- Pygame script -------------------------------

import pygame

SCREEN_DIM = 1920, 1080
MARGINS = 100, 100

x_step = (SCREEN_DIM[0] - 2 * MARGINS[0]) / 8
y_step = (SCREEN_DIM[1] - 2 * MARGINS[1]) / 8


def draw_move_lines(screen, move_dict, origin=None):
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
