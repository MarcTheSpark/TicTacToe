import pygame
import numpy as np
from boardstates import BoardStatesList
from utils import square_format


pygame.init()

WIDTH, HEIGHT = 1920, 1080

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("Tic Tac Toe States")

clock = pygame.time.Clock()

LINE_WIDTH = 12
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
O_COLOR = (0, 255, 0)
X_COLOR = (255, 150, 50)


def draw_board(x0, y0, width, height):
    square_size = width // BOARD_COLS
    screen.fill(BLACK, rect=(x0, y0, width, height))
    for row in range(1, BOARD_ROWS):
        pygame.draw.line(screen, WHITE, (x0, row * square_size), (width, row * square_size), LINE_WIDTH)
    for col in range(1, BOARD_COLS):
        pygame.draw.line(screen, WHITE, (col * square_size, y0), (col * square_size, height), LINE_WIDTH)


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


def calculate_image_positions(canvas_width, canvas_height, num_images, overall_padding, inter_rect_padding_percent):
    """
    From ChatGPT. overall_padding is in px, inter_rect_padding_percent is in % of rect_width
    """
    if not hasattr(overall_padding, '__len__'):
        overall_padding = (overall_padding, overall_padding)

    if len(overall_padding) < 4:
        overall_padding *= 2

    # Adjust canvas size for overall padding
    adj_width = canvas_width - overall_padding[0] - overall_padding[2]
    adj_height = canvas_height - overall_padding[1] - overall_padding[3]

    # Find the best fit for rows and columns
    best_layout = (0, 0)
    max_size = 0
    for rows in range(1, num_images + 1):
        cols = -(-num_images // rows)  # Ceiling division
        if rows * cols >= num_images:
            # Calculate size considering inter-rectangle padding
            size = min(adj_width // cols,
                       adj_height // rows)
            if size > max_size:
                max_size = size
                best_layout = (rows, cols)

    rows, cols = best_layout
    inter_rect_padding = inter_rect_padding_percent * max_size
    size = min((adj_width - (cols - 1) * inter_rect_padding) // cols,
               (adj_height - (rows - 1) * inter_rect_padding) // rows)

    # Centering adjustment
    total_width = cols * size + (cols - 1) * inter_rect_padding
    total_height = rows * size + (rows - 1) * inter_rect_padding
    start_x = overall_padding[0] + (adj_width - total_width) // 2
    start_y = overall_padding[1] + (adj_height - total_height) // 2

    # Generate coordinates for each image
    positions = []
    for i in range(num_images):
        row = i // cols
        col = i % cols
        x = start_x + col * (size + inter_rect_padding)
        y = start_y + row * (size + inter_rect_padding)
        positions.append((x, y, size, size))

    return positions


font = pygame.font.Font(None, 80)


def draw_states(which_move):
    global LINE_WIDTH
    screen.fill((0, 0, 0))
    states_to_draw = square_format(all_board_states.board_states[which_move])

    board_rects = calculate_image_positions(WIDTH, HEIGHT, len(states_to_draw), (150, 120, 150, 40), 0.1)
    LINE_WIDTH = max(1, int(12 * board_rects[0][3] / HEIGHT))

    text = font.render(f"Move {current_state}         {len(all_board_states.board_states[current_state])} state" + ("s" if len(all_board_states.board_states[current_state]) > 1 else ""), True, (255, 255, 255))
    screen.blit(text, ((WIDTH  - text.get_width()) / 2, 30))

    for board_state, rect in zip(states_to_draw, board_rects):
        draw_board(*rect)
        winning_line, winning_player = check_winner(board_state)
        draw_xo(board_state, *rect, winning_line, winning_player)
        pygame.display.update()


all_board_states = BoardStatesList.complete().standard_forms_only()
current_state = 0
draw_states(current_state)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Check for keypress event
        if event.type == pygame.KEYDOWN:
            # Here you can check which key was pressed
            if event.key == pygame.K_ESCAPE:
                running = False  # Exit the loop if ESC is pressed
            elif event.key == pygame.K_SPACE:
                current_state += 1
                try:
                    draw_states(current_state)
                except IndexError:
                    running = False
                    break
            else:
                print(f"Key {pygame.key.name(event.key)} pressed")

clock.tick(60)
