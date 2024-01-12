import pygame
import numpy as np
import time
from specific_games import start_with_center, start_game
from game_extrapolation import extrapolate_all_games, remove_near_duplicates
from scamp import Session, wait, wait_for_children_to_finish


the_20_games = extrapolate_all_games([start_with_center], skill=4)
the_14_games = remove_near_duplicates(the_20_games)
games = the_14_games

# # One random game
# game = np.array(
#     [[[0, 0, 0],
#       [0, 0, 0],
#       [0, 0, 0]],
#      [[0, 0, 0],
#       [1, 0, 0],
#       [0, 0, 0]],
#      [[0, 0, 0],
#       [1, -1, 0],
#       [0, 0, 0]],
#      [[0, 0, 0],
#       [1, -1, 0],
#       [0, 0, 1]],
#      [[0, -1, 0],
#       [1, -1, 0],
#       [0, 0, 1]],
#      [[0, -1, 0],
#       [1, -1, 0],
#       [0, 1, 1]],
#      [[0, -1, 0],
#       [1, -1, 0],
#       [-1, 1, 1]],
#      [[0, -1, 0],
#       [1, -1, 0],
#       [-1, 1, 1]]]
# )
# game_iter = iter([game])


# # tons of games in random order
# all_games = extrapolate_all_games([start_game], skill=2)
# import random
# random.shuffle(all_games)
# game_iter = iter(all_games)


# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 900, 900
LINE_WIDTH = 10
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
O_COLOR = (0, 255, 0)
X_COLOR = (255, 150, 50)

# Set up the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe Animation")


def draw_board(x0, y0, width, height):
    square_size = width // BOARD_COLS
    screen.fill(BLACK, rect=(x0, y0, width, height))
    for row in range(1, BOARD_ROWS):
        pygame.draw.line(screen, WHITE, (x0, row * square_size), (width, row * square_size), LINE_WIDTH)
    for col in range(1, BOARD_COLS):
        pygame.draw.line(screen, WHITE, (col * square_size, y0), (col * square_size, height), LINE_WIDTH)


def draw_xo(game_state, x0, y0, width, height, winning_line=None, winning_player=None):
    square_size = width // BOARD_COLS
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            center = (int(col * square_size + square_size // 2), int(row * square_size + square_size // 2))
            if game_state[row][col] == -1:
                color = O_COLOR if (winning_player == -1 and is_winning_square(row, col, winning_line)) else WHITE
                pygame.draw.circle(screen, color, center, square_size * 0.4, LINE_WIDTH)
            elif game_state[row][col] == 1:
                color = X_COLOR if (winning_player == 1 and is_winning_square(row, col, winning_line)) else WHITE
                margin = square_size * 0.1
                pygame.draw.line(screen, color, (col * square_size + margin, row * square_size + margin),
                                 (col * square_size + square_size - margin,
                                  row * square_size + square_size - margin), int(LINE_WIDTH * 1.2))
                pygame.draw.line(screen, color,
                                 (col * square_size + margin, row * square_size + square_size - margin),
                                 (col * square_size + square_size - margin, row * square_size + margin),
                                 int(LINE_WIDTH * 1.2))


def check_winner(game_state):
    """
    Returns the coordinates of the start and end of the winning line, or (None, None) if no winning line
    Kinda a weird way of formatting this information?
    """
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
    """
    Checks if this square is on the given winning line.
    """
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


s = Session().run_as_server()
s.print_available_midi_output_devices()
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


def animate_game(all_games, frame_dur):
    last_state = None
    for game_state in all_games:
        winning_line, winning_player = check_winner(game_state)
        if last_state is not None:
            delta = game_state - last_state
            coords = list(zip(*np.where(delta != 0)))[0]
            if winning_line:
                inst = ohs_long if np.sum(delta) == -1 else exes_long
                coords1, coords3 = winning_line
                coords2 = ((coords1[0] + coords3[0]) / 2, (coords1[1] + coords3[1]) / 2)
                chord = [coords_to_pitch(coord) for coord in (coords1, coords2, coords3)]
                inst.play_chord(chord, [0.2, 1.0], frame_dur/3)
                # s.fork(roll_chord, (inst, chord, 0.7))
            else:
                inst = ohs if np.sum(delta) == -1 else exes
                inst.play_note(coords_to_pitch(coords), 0.6, frame_dur / 3)
        last_state = game_state
        draw_board(0, 0, WIDTH, HEIGHT)
        draw_xo(game_state, 0, 0, WIDTH, HEIGHT, winning_line, winning_player)
        pygame.display.update()
        time.sleep(frame_dur)


for game in games:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break

    # Start the animation (adjust frame_dur as needed)
    animate_game(game, frame_dur=0.2)
    time.sleep(0.6)

pygame.quit()
