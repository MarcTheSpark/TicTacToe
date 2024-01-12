import dataclasses

from boardstates import BoardStatesList, get_standard_form
from game_extrapolation import (extrapolate_all_games, remove_near_duplicates, start_game, start_with_center,
                                check_win_status)
from utils import tuple_format, square_format, get_move_sequence
import numpy as np
import time


FRAME_DUR = 0.18

games = [tuple_format(game) for game in extrapolate_all_games([start_game], skill=7)]
# Start with center truncated vs non-truncated is very interesting/revealing

used_states = BoardStatesList.from_games(games)
used_states_standard_forms = used_states.standard_forms_only()

games_state_indices = [tuple(used_states_standard_forms.get_state_index(get_standard_form(board_state))[1]
                             for board_state in game)
                       for game in games]

highlighted_games = [-1]

# ---------------- Filter down games by paring down games that have the same standardized index sequence -------------

pared_down_indices = []
pared_down_games = []
index_set = set()

for game, state_index in zip(games, games_state_indices):
    if state_index not in index_set:
        pared_down_indices.append(state_index)
        pared_down_games.append(game)
        index_set.add(state_index)

games_state_indices = pared_down_indices
games = pared_down_games

# --------------------------------- win statuses and game moves --------------------------------------

game_win_statuses = [check_win_status(game) for game in games]
game_move_sequences = [get_move_sequence(game) for game in games]


# --------------------------------------- truncate ---------------------------------------------

def truncate_sequences(game_sequences, game_results):
    prefix_map = {}

    # Step 1: Create initial prefix map
    for sequence, result in zip(game_sequences, game_results):
        for i in range(1, len(sequence) + 1):
            prefix = sequence[:i]
            if prefix in prefix_map:
                prefix_map[prefix].add(result)
            else:
                prefix_map[prefix] = {result}

    # Step 2: Truncate sequences
    truncated_sequences = {}
    for sequence, result in zip(game_sequences, game_results):
        for i in range(1, len(sequence) + 1):
            prefix = sequence[:i]
            if len(prefix_map[prefix]) == 1 and result in prefix_map[prefix]:
                truncated_sequences[prefix] = result
                break

    return truncated_sequences


truncated_sequence_dict = truncate_sequences(games_state_indices, game_win_statuses)

truncated_game_state_indices = list(truncated_sequence_dict.keys())
truncated_game_win_statuses = list(truncated_sequence_dict.values())
truncated_games = [used_states_standard_forms.index_sequence_to_board_sequence(truncated_game_state_index_sequence)
                   for truncated_game_state_index_sequence in truncated_game_state_indices]

original_game_stata_indices = games_state_indices
original_game_win_statuses = game_win_statuses
original_games = games

# ------------------------------ pygame ----------------------------------

import pygame
import pygame.draw

# Initialize Pygame
pygame.init()

# Get the screen resolution of your monitor
infoObject = pygame.display.Info()
SCREEN_WIDTH, SCREEN_HEIGHT = infoObject.current_w, infoObject.current_h

WORLD_WIDTH, WORLD_HEIGHT = 16, 9
CELL_SIZE = SCREEN_WIDTH / 16

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

# ------------------ game animation -------------------------

ORIGIN = 60, SCREEN_HEIGHT * 0.65 - 60
WIDTH, HEIGHT = SCREEN_HEIGHT * 0.35, SCREEN_HEIGHT * 0.35
LINE_WIDTH = 5
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
O_COLOR = (0, 255, 0)
X_COLOR = (255, 150, 50)


def draw_board(x0, y0, width, height):
    square_size = width // BOARD_COLS
    screen.fill((0, 0, 0), rect=(x0, y0, width, height))
    for row in range(1, BOARD_ROWS):
        pygame.draw.line(screen, WHITE, (x0, y0 + row * square_size), (x0 + width, y0 + row * square_size), LINE_WIDTH)
    for col in range(1, BOARD_COLS):
        pygame.draw.line(screen, WHITE, (x0 + col * square_size, y0), (x0 + col * square_size, y0 + height), LINE_WIDTH)


def draw_xo(game_state, x0, y0, width, height, winning_line=None, winning_player=None):
    square_size = width // BOARD_COLS
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            center = (int(x0 + col * square_size + square_size // 2), int(y0 + row * square_size + square_size // 2))
            if game_state[row][col] == -1:
                color = O_COLOR if (winning_player == -1 and is_winning_square(row, col, winning_line)) else WHITE
                pygame.draw.circle(screen, color, center, square_size * 0.4, LINE_WIDTH)
            elif game_state[row][col] == 1:
                color = X_COLOR if (winning_player == 1 and is_winning_square(row, col, winning_line)) else WHITE
                margin = square_size * 0.1
                pygame.draw.line(screen, color, (x0 + col * square_size + margin, y0 + row * square_size + margin),
                                 (x0 + col * square_size + square_size - margin,
                                  y0 + row * square_size + square_size - margin), int(LINE_WIDTH * 1.2))
                pygame.draw.line(screen, color,
                                 (x0 + col * square_size + margin, y0 + row * square_size + square_size - margin),
                                 (x0 + col * square_size + square_size - margin, y0 + row * square_size + margin),
                                 int(LINE_WIDTH * 1.2))


def check_winner(game_state):
    # Check rows and columns
    for i in range(3):
        if abs(game_state[i, :].sum()) == 3:  # Check rows
            return ((i, 0), (i, 2)), game_state[i, 0]
        if abs(game_state[:, i].sum()) == 3:  # Check columns
            return ((0, i), (2, i)), game_state[0, i]

    # Check diagonals
    if abs(np.diag(game_state).sum()) == 3:
        return ((0, 0), (2, 2)), game_state[0, 0]
    if abs(np.diag(np.fliplr(game_state)).sum()) == 3:
        return ((0, 2), (2, 0)), game_state[0, 2]

    return None, None  # No winner


def is_winning_square(row, col, winning_line):
    if not winning_line:
        return False
    ((start_row, start_col), (end_row, end_col)) = winning_line
    if start_row == end_row:  # Winning row
        return row == start_row
    elif start_col == end_col:  # Winning column
        return col == start_col
    elif start_row < end_row and start_col < end_col:  # Diagonal from top-left to bottom-right
        return row == col
    else:  # Diagonal from top-right to bottom-left
        return row + col == 2


from scamp import *

s = Session().run_as_server()
ohs = s.new_midi_part("midi through Port-0")
exes = s.new_midi_part("midi through Port-1")
ohs_long = s.new_midi_part("midi through Port-2")
exes_long = s.new_midi_part("midi through Port-3")


def coords_to_pitch(coords):
    return 70 - 4.98 * coords[1] + 3.86 * coords[0]


def roll_chord(inst, pitches, volume=1, spacing=0.07, length=1):
    length_left = length
    for pitch in pitches:
        inst.play_note(pitch, volume, length_left, blocking=False)
        wait(spacing)
        length_left -= spacing
    wait_for_children_to_finish()


@dataclasses.dataclass
class GameAnimation:
    game_state: np.ndarray
    winning_line: tuple
    winning_player: int

    def draw(self):
        draw_board(*ORIGIN, WIDTH, HEIGHT)
        draw_xo(self.game_state, *ORIGIN, WIDTH, HEIGHT, self.winning_line, self.winning_player)

        i, game_state_index = used_states_standard_forms.get_state_index(get_standard_form(self.game_state))
        num_states_this_move = used_states_standard_forms.sizes_per_move()[i]
        x_step, y_step = DRAW_BOX[2] / 9, DRAW_BOX[3] / num_states_this_move
        this_point = DRAW_BOX[0] + i * x_step, DRAW_BOX[1] + game_state_index * y_step
        world_ellipse(screen, (255, 255, 0), (this_point[0] - 0.07, this_point[1] - 0.07, 0.14, 0.14))


game_animation: GameAnimation = None


def animate_game(game, frame_dur):
    global game_animation
    last_state = None
    for game_state in game:
        winning_line, winning_player = check_winner(game_state)
        if last_state is not None:
            delta = game_state - last_state
            coords = list(zip(*np.where(delta != 0)))[0]
            if winning_line:
                inst = ohs_long if np.sum(delta) == -1 else exes_long
                coords1, coords3 = winning_line
                coords2 = ((coords1[0] + coords3[0]) / 2, (coords1[1] + coords3[1]) / 2)
                chord = [coords_to_pitch(coord) for coord in (coords1, coords2, coords3)]
                inst.play_chord(chord, [0.2, 1.0], frame_dur / 3)
            else:
                inst = ohs if np.sum(delta) == -1 else exes
                inst.play_note(coords_to_pitch(coords), 0.6, frame_dur / 3)
            game_animation = GameAnimation(game_state, winning_line, winning_player)
        last_state = game_state

        wait(frame_dur)
    wait(frame_dur * 2)
    game_animation = None


# ------------------- line drawings ------------------------

DRAW_BOX = (1, 0.5, 14, 8)


def draw_game_state_sequence(state_sequence, surface, color=(255, 255, 255), cap_color=(255, 255, 255), width=0.02):
    last_point = None
    x_step = DRAW_BOX[2] / 9
    for i, (game_state_index, num_states_this_move) in enumerate(zip(state_sequence, used_states_standard_forms.sizes_per_move())):
        y_step = DRAW_BOX[3] / num_states_this_move
        this_point = DRAW_BOX[0] + i * x_step, DRAW_BOX[1] + game_state_index * y_step
        if last_point:
            world_line(surface, color, last_point, this_point, width=width)
        world_ellipse(surface, color, (this_point[0] - 0.03, this_point[1] - 0.03, 0.06, 0.06))
        last_point = this_point
    world_ellipse(surface, cap_color, (this_point[0] - 0.05, this_point[1] - 0.05, 0.1, 0.1))


def draw_games(screen):
    highlights = []
    for i, (this_game_indices, win_status) in enumerate(zip(games_state_indices, game_win_statuses)):
        cap_color = (255, 150, 50) if win_status == 1 else \
            (0, 255, 0) if win_status == -1 else \
            (255, 255, 255)
        if i in highlighted_games:
            highlights.append((i, (this_game_indices, win_status)))
            continue
        draw_game_state_sequence(this_game_indices, screen, cap_color=cap_color, width=0.005)

    for i, (this_game_indices, win_status) in highlights:
        cap_color = (255, 150, 50) if win_status == 1 else \
            (0, 255, 0) if win_status == -1 else \
                (255, 255, 255)
        draw_game_state_sequence(this_game_indices, screen, color=(255, 255, 0), cap_color=cap_color, width=0.02)


# ------------------- pygame main ---------------------------

clock = pygame.time.Clock()

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)

KEY_REPEAT_DELAY = 500
KEY_REPEAT_INTERVAL = 100
key_repeat_countdown = None

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

    dt = clock.tick(60)

    keys = pygame.key.get_pressed()

    if not any(pygame.key.get_pressed()):
        key_repeat_countdown = None
    elif key_repeat_countdown is None or key_repeat_countdown < 0:
        if keys[pygame.K_UP]:
            highlighted_games[0] = len(games) - 1
        elif keys[pygame.K_DOWN]:
            highlighted_games[0] = 0
        elif keys[pygame.K_LEFT]:
            highlighted_games[0] = max(highlighted_games[0] - 1, -1)
        elif keys[pygame.K_RIGHT]:
            highlighted_games[0] = min(highlighted_games[0] + 1, len(games) - 1)
        elif keys[pygame.K_t]:
            if games_state_indices == original_game_stata_indices:
                games_state_indices = truncated_game_state_indices
                game_win_statuses = truncated_game_win_statuses
                games = truncated_games
                highlighted_games = [-1]
            else:
                games_state_indices = original_game_stata_indices
                game_win_statuses = original_game_win_statuses
                games = original_games
                highlighted_games = [-1]

        elif keys[pygame.K_SPACE]:
            s.fork(animate_game, (square_format(games[highlighted_games[0]]), FRAME_DUR))
        key_repeat_countdown = KEY_REPEAT_DELAY if key_repeat_countdown is None else KEY_REPEAT_INTERVAL

    if key_repeat_countdown is not None:
        key_repeat_countdown -= dt

    # Clear screen
    screen.fill((0, 0, 0))

    # draw_games_state_tree(screen)
    draw_games(screen)

    if game_animation:
        game_animation.draw()

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
