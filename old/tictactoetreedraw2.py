# Pointing out that the branches are deceptive, because they don't represent the same board state
# FOR EACH MOVE, MAKE Y Axis every possible board state after those moves.
# That way we can see forking and branching paths


import json

def convert_keys_to_int(x):
    """Recursively converts string keys to integers in a nested dictionary."""
    if isinstance(x, dict):
        return {int(k): convert_keys_to_int(v) for k, v in x.items()}
    return x

with open('movesDict.json', 'r') as file:
    all_moves = convert_keys_to_int(json.load(file))

# print(all_moves)
# def all_nested_moves(moves_available):
#     return {x: all_nested_moves(tuple(y for y in moves_available if y != x)) for x in moves_available}
#
# all_moves = all_nested_moves(range(1, 10))


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