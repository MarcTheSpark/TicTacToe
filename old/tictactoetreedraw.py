all_moves = {5: {2: {7: {3: {1: {9: {4: {}}}}}, 4: {6: {7: {3: {1: {}}}, 1: {9: {7: {}}}}}, 1: {9: {7: {4: {3: {}}}, 4: {7: {6: {}}}}}}, 1: {2: {8: {3: {7: {9: {4: {}}}}, 7: {3: {6: {4: {9: {}}}, 4: {6: {9: {}}}}}, 6: {4: {7: {3: {9: {}}}}}, 4: {6: {7: {3: {9: {}}}, 3: {7: {9: {}}}}}}}, 6: {4: {7: {3: {2: {8: {9: {}}}}}}}, 3: {7: {4: {6: {8: {2: {9: {}}}, 2: {8: {9: {}}}}}}}}}}


def all_nested_moves(moves_available):
    return {x: all_nested_moves(moves_available - {x}) for x in moves_available}


# all_moves = all_nested_moves(set(range(1, 10)))

import pygame

SCREEN_DIM = 1920, 1080
MARGINS = 20, 20

x_step = (SCREEN_DIM[0] - 2 * MARGINS[0]) / 9
start_origin = MARGINS[0], SCREEN_DIM[1] / 2
start_vertical_range = 0.8 * SCREEN_DIM[1] - 2 * MARGINS[1]

def draw_move_lines(screen, move_dict, origin, vertical_range):
    vertical_step = vertical_range / 9
    for x in move_dict:
        endpoint = origin[0] + x_step, origin[1]  + (x - 5) * vertical_step
        pygame.draw.line(screen, (255, 255, 255), origin, endpoint, 2)
        
        if len(move_dict[x]) > 0:
            draw_move_lines(screen, move_dict[x], endpoint, vertical_range / 3)

            
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
        draw_move_lines(screen, all_moves, start_origin, start_vertical_range)
                
        # Update the display
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()