import numpy as np

# ------------------------- board states ----------------------------

from boardstates import all_board_states
from tictactoe import extrapolate_all_games, remove_near_duplicates, start_game, start_with_center, check_win_status

def get_board_state_index(board_state, move_num):
    if not isinstance(board_state, np.ndarray):
        board_state = np.array(board_state)
    this_move_states = all_board_states[move_num]
    
    matches = np.where(np.all(this_move_states == board_state, axis=1))[0]
    if len(matches) > 0:
        return matches[0]
    else:
        return None

games = extrapolate_all_games([start_game], skill=3)
#games = remove_near_duplicates(games)
game_win_statuses = [check_win_status(game) for game in games]
games = [game.reshape((-1, 9)) for game in games]


# ---------------------------- filter out which states are actually used -----------------

used_states = [[] for _ in range(10)]
for game in games:
    for i, state in enumerate(game):
        used_states[i].append(get_board_state_index(state, i))


all_board_states = [board_states[sorted(set(used_states[i]))] for i, board_states in enumerate(all_board_states)]


# ------------------------- pygame setup ----------------------------

import pygame
import pygame.draw

# Initialize Pygame
pygame.init()

# Constants for screen and world dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
WORLD_WIDTH, WORLD_HEIGHT = 16, 9
CELL_SIZE = 1920/16

# Zoom and pan variables
zoom = 1
pan_x = 0
pan_y = 0


# --------------------- world drawing utilities ---------------------

# Function to convert world coordinates to screen coordinates
def world_to_screen(x, y):
    screen_x = x * CELL_SIZE * zoom + pan_x
    screen_y = y * CELL_SIZE * zoom + pan_y
    return int(screen_x), int(screen_y)

def world_to_screen_width(width):
    return max(1, int(width * CELL_SIZE * zoom))

# Wrapper for rect
def world_rect(surface, color, world_rect, width=0, **kwargs):
    screen_rect = pygame.Rect(
        world_to_screen(world_rect[0], world_rect[1]),
        (world_rect[2] * CELL_SIZE * zoom, world_rect[3] * CELL_SIZE * zoom)
    )
    return pygame.draw.rect(surface, color, screen_rect, world_to_screen_width(width), **kwargs)

# Wrapper for line
def world_line(surface, color, start_pos, end_pos, width=1):
    screen_start = world_to_screen(*start_pos)
    screen_end = world_to_screen(*end_pos)
    return pygame.draw.line(surface, color, screen_start, screen_end, world_to_screen_width(width))

# Wrapper for circle
def world_circle(surface, color, center, radius, width=0, **kwargs):
    screen_center = world_to_screen(*center)
    screen_radius = radius * CELL_SIZE * zoom
    return pygame.draw.circle(surface, color, screen_center, screen_radius, world_to_screen_width(width), **kwargs)

# Wrapper for ellipse
def world_ellipse(surface, color, world_rect, width=0):
    # Convert the world rectangle to screen rectangle
    screen_rect = pygame.Rect(
        world_to_screen(world_rect[0], world_rect[1]),
        (world_rect[2] * CELL_SIZE * zoom, world_rect[3] * CELL_SIZE * zoom)
    )
    return pygame.draw.ellipse(surface, color, screen_rect, width)

# Wrapper for polygon
def world_polygon(surface, color, points, width=0, **kwargs):
    screen_points = [world_to_screen(*point) for point in points]
    return pygame.draw.polygon(surface, color, screen_points, world_to_screen_width(width), **kwargs)


def draw_game(game, draw_box, surface, color=(255, 255, 255), cap_color=(255, 255, 255), width=0.02):
    x_step = draw_box[2] / 9
    last_point = None
    for i, board_state in enumerate(game):
        y_step = draw_box[3] / len(all_board_states[i])
        this_point = draw_box[0] + i * x_step, draw_box[1] + get_board_state_index(board_state, i) * y_step
        if last_point:
            world_line(surface, color, last_point, this_point, width=width)
        world_ellipse(surface, color, (this_point[0] - 0.03, this_point[1] - 0.03, 0.06, 0.06))
        last_point = this_point
    world_ellipse(surface, cap_color, (this_point[0] - 0.03, this_point[1] - 0.03, 0.06, 0.06))
    

# ------------------- pygame main ---------------------------


# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEWHEEL:
            if pygame.key.get_mods() & pygame.KMOD_CTRL:
                # Capture the current mouse position
                mouse_x, mouse_y = pygame.mouse.get_pos()

                # Convert mouse position to world coordinates before zoom
                world_mouse_x_before = (mouse_x - pan_x) / (CELL_SIZE * zoom)
                world_mouse_y_before = (mouse_y - pan_y) / (CELL_SIZE * zoom)

                # Calculate the zoom factor
                old_zoom = zoom
                zoom *= 1 + event.y * (0.09 if pygame.key.get_mods() & pygame.KMOD_SHIFT else 0.03)
                zoom = max(0.01, min(zoom, 50))

                # Convert mouse position to world coordinates after zoom
                world_mouse_x_after = (mouse_x - pan_x) / (CELL_SIZE * zoom)
                world_mouse_y_after = (mouse_y - pan_y) / (CELL_SIZE * zoom)

                # Adjust pan to keep the mouse position constant in world coordinates
                pan_x += (world_mouse_x_after - world_mouse_x_before) * CELL_SIZE * zoom
                pan_y += (world_mouse_y_after - world_mouse_y_before) * CELL_SIZE * zoom
            else:
                pan_x += -event.x * 20 * zoom ** 0.5
                pan_y += event.y * 20 * zoom ** 0.5

    # Clear screen
    screen.fill((0, 0, 0))

    for game, win_status in zip(games, game_win_statuses):
        cap_color = (255, 100, 0) if win_status == 1 else \
                    (0, 100, 255) if win_status == -1 else \
                    (150, 150, 150)
        draw_game(game, (1, 0.5, 14, 8), screen, cap_color=cap_color, width=0.005)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
