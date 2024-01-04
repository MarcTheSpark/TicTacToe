"""There are exactly 20 different* games of Tic-Tac-Toe"""

import numpy as np
from minimaz import get_best_moves


def are_symmetrical(game1, game2):
    """
    Check if two Tic-Tac-Toe games are rotationally or mirror symmetrical.

    :param game1: A 3D NumPy array representing the first Tic-Tac-Toe game (nx3x3).
    :param game2: A 3D NumPy array representing the second Tic-Tac-Toe game (nx3x3).
    :return: True if the games are symmetrical, False otherwise.
    """

    # Check if game lengths are different
    if game1.shape[0] != game2.shape[0]:
        return False

    # Check for direct, rotational, and mirror symmetry
    for k in range(0, 4):  # Rotations of 90, 180, 270 degrees
        rotated_game2 = np.rot90(game2, k, axes=(1, 2))
        if np.array_equal(game1, rotated_game2) or \
           np.array_equal(game1, np.flip(rotated_game2, 2)):
            return True

    return False


def is_valid_game(game):
    """
    Check if a Tic-Tac-Toe game is valid. (Probably not perfect.)

    :param game: A 3D NumPy array representing the Tic-Tac-Toe game (nx3x3).
    :return: True if the game is valid, False otherwise.
    """

    # Start with an empty board
    if not np.array_equal(game[0], np.zeros((3, 3))):
        return False

    for i in range(1, game.shape[0]):
        # Check the sum for odd and even turns
        if np.sum(game[i]) != i % 2:
            return False

        # Check that there is exactly one difference from the previous turn
        if np.sum(np.abs(game[i] - game[i-1])) != 1:
            return False

    return True


def check_win_status(game):
    """
    Check if a Tic-Tac-Toe game is over and if so, who won

    :param game: A 3D NumPy array representing a Tic-Tac-Toe game (nx3x3).
    :return: -1 if X's win, 1 if O's win, and 0 if draw; False if not over
    """
    last_board = game[-1]
    if check_win(last_board, 1):
        return 1  # O wins
    elif check_win(last_board, -1):
        return -1  # X wins
    elif np.sum(np.abs(last_board)) == 9:
        return 0  # draw
    else:
        return None  # unfinished


def check_win(board, player):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if all(board[i, :] == player) or all(board[:, i] == player):
            return True
    if all(np.diag(board) == player) or all(np.diag(np.fliplr(board)) == player):
        return True
    return False


def find_open_locations(board):
    """Just get all the zeros"""
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]


def find_winning_locations(board):
    """
    Identify all O-win-locations and X-win-locations on a Tic-Tac-Toe board.

    :param board: A 3x3 NumPy array representing the Tic-Tac-Toe board.
    :return: Two lists containing the coordinates of winning locations for O and X.
    """
    o_wins = []
    x_wins = []

    for i, j in find_open_locations(board):
        if board[i, j] == 0:
            # Check for O win
            temp_board = np.copy(board)
            temp_board[i, j] = 1  # Place an O
            if check_win(temp_board, 1):
                o_wins.append((i, j))

            # Check for X win
            temp_board[i, j] = -1  # Place an X
            if check_win(temp_board, -1):
                x_wins.append((i, j))

    return o_wins, x_wins


def find_win_setup_locations(board, player, only_the_best=False):
    setup_to_win = []
    threshold = 1
    for i, j in find_open_locations(board):
        # Check for O win
        temp_board = np.copy(board)
        temp_board[i, j] = player  # Place an O
        
        index = 0 if player == 1 else 1
        num_win_locations = len(find_winning_locations(temp_board)[index])
        if num_win_locations >= threshold:
            if only_the_best and num_win_locations > threshold:
                threshold = num_win_locations  # needs to be at least as good as the next best one
                setup_to_win.clear()
            setup_to_win.append((i, j))
    return setup_to_win


def is_game_over(game):
    """
    Check if a Tic-Tac-Toe game is over by examining the last frame.

    :param game: A 3D NumPy array representing a Tic-Tac-Toe game (nx3x3).
    :return: True if the game is over (win or draw), False otherwise.
    """
    return check_win_status(game) is not None


def extrapolate_one_step(game, skill=0):
    """
    Extrapolates next possible moves. If skill is 0, then any move is considered.
    If skill is 1, then we will at least take a win if it's available on our turn.
    If skill is 2, then we will avoid, if possible, giving our opponent a winning move.
    If skill is 3, then we try to set up a 2 in a row, if possible
    If skill is 4, then we try to do a fork if possible and if not try for 2 in a row
    If skill is 5, then we do any move that is included in skill=3 or a minimax move
    If skill is 6, then we do any move that is included in skill=4 or a minimax move
    If skill is 7, then we do pure minimax play
    """
    extrapolated = []
    # Determine whose turn it is (1 for O's turn, -1 for X's turn)
    current_board = game[-1]
    whose_turn = 1 if np.sum(current_board) % 2 == 0 else -1
    
    winning_moves = find_winning_locations(current_board)
    player_wins, opponent_wins = winning_moves if whose_turn == 1 else winning_moves[::-1]
    
    if skill >= 1 and len(player_wins) > 0:
        next_moves = player_wins
    elif skill >= 2 and len(opponent_wins) > 0:
        next_moves = opponent_wins
    elif skill >= 3:
        # using more "sophisticated" strategies
        if skill >= 7:
            # Perfect (minimax) skill (what does this mean exactly)
            next_moves = get_best_moves(current_board)
        elif skill >= 6:
            # Minimax
            minimax_next_moves = get_best_moves(current_board)
            # Heuristic (fork if we can)
            good_plays = find_win_setup_locations(current_board, whose_turn, only_the_best=True)
            # mix of minimax and heuristic
            next_moves = sorted(set(good_plays + minimax_next_moves))
        elif skill >= 5:
            # Minimax
            minimax_next_moves = get_best_moves(current_board)
            # Heuristic (fork if we can)
            good_plays = find_win_setup_locations(current_board, whose_turn, only_the_best=False)
            # mix of minimax and heuristic
            next_moves = sorted(set(good_plays + minimax_next_moves))
        else:
            good_plays = find_win_setup_locations(current_board, whose_turn, only_the_best=skill >= 4)
            if len(good_plays) > 0:
                next_moves = good_plays
            else:
                next_moves = find_open_locations(current_board)
    else:
        next_moves = find_open_locations(current_board)
        

    for i, j in next_moves:
        # Create a new board with the current move
        new_board = np.copy(current_board)
        new_board[i, j] = whose_turn
        # Create a new game state by adding the new board to the game history
        new_game = np.concatenate((game, [new_board]), axis=0)
        extrapolated.append(new_game)
    return extrapolated


def prune_symmetrical_games(games_list):
    to_remove = set()  # A set to keep track of indices of games to be removed
    n = len(games_list)

    for i in range(n):
        if i in to_remove:
            continue  # Skip if this game is already marked for removal

        for j in range(i + 1, n):
            if j in to_remove:
                continue  # Skip if the other game is already marked for removal

            if are_symmetrical(games_list[i], games_list[j]):
                to_remove.add(j)  # Mark the symmetrical game for removal

    # Removing the marked games
    # Since removing by index can be tricky (as the list size changes), we'll do it in a reversed order
    for index in sorted(to_remove, reverse=True):
        del games_list[index]

    return games_list


def extrapolate_all_games(unfinished_games, skill=0, prune_symmetrical=True, show_log=True):
    
    finished_games = []
    
    for game in reversed(unfinished_games):
        if is_game_over(game):
            unfinished_games.remove(game)
            finished_games.append(game)
                
    while len(unfinished_games) > 0:
        if show_log:
            print(f"PRE: {len(unfinished_games)}, {len(finished_games)}")
        unfinished_games = [
            extrapolated_game
            for game in unfinished_games
            for extrapolated_game in
            (prune_symmetrical_games(extrapolate_one_step(game, skill=skill))
             if prune_symmetrical else extrapolate_one_step(game, skill=skill)) 
        ]
        if show_log:
            print(f"EXTRAPOLATE: {len(unfinished_games)}, {len(finished_games)}")
        for i in reversed(range(len(unfinished_games))):
            if is_game_over(unfinished_games[i]):
                finished_games.append(unfinished_games[i])
                del unfinished_games[i]
        if show_log:
            print(f"PRUNE ENDED: {len(unfinished_games)}, {len(finished_games)}")
    return finished_games


def remove_near_duplicates(games):
    """
    Remove games that are duplicates up until the last two moves.

    :param games: A list of 3D NumPy arrays representing Tic-Tac-Toe games.
    :return: A list of games with near duplicates removed.
    """
    indices_to_remove = set()

    for i in range(len(games)):
        if i in indices_to_remove:
            continue  # Skip if this game is already marked for removal

        for j in range(i + 1, len(games)):
            if j in indices_to_remove:
                continue  # Skip if the other game is already marked for removal

            # Compare the games up to the penultimate move
            game_i_slice = games[i][:-2, :, :]
            game_j_slice = games[j][:-2, :, :]

            if check_win_status(games[i]) == check_win_status(games[j]) and np.array_equal(game_i_slice, game_j_slice):
                indices_to_remove.add(j)  # Mark the duplicate game for removal

    # Create a new list of unique games
    unique_games = [games[i] for i in range(len(games)) if i not in indices_to_remove]

    return unique_games

       
       
start_game = np.array(
    [[[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]]]
)

start_with_center = np.array(
    [[[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]],
     [[0, 0, 0],
      [0, 1, 0],
      [0, 0, 0]]]
)

start_with_edge = np.array(
    [[[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]],
     [[0, 0, 0],
      [1, 0, 0],
      [0, 0, 0]]]
)

start_with_corner = np.array(
    [[[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]],
     [[1, 0, 0],
      [0, 0, 0],
      [0, 0, 0]]]
)

start_with_corner_middle = np.array(
    [[[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]],
     [[1, 0, 0],
      [0, 0, 0],
      [0, 0, 0]],
     [[1, 0, 0],
      [0, -1, 0],
      [0, 0, 0]]]
)

# 9! = 362880, but this doesn't account for games that end early
# print(len(extrapolate_all_games([start_game], prune_symmetrical=False)), "games in total")  # 255168
# print(len(extrapolate_all_games([start_game])), "games in total")  # 31896
# print(len(extrapolate_all_games([start_game], skill=1)), "games with idiotic moves removed")  # 6956
# print(len(extrapolate_all_games([start_game], skill=2)), "games with dumb moves removed")  # 2936
# print(len(extrapolate_all_games([start_game], skill=3)), "games with decent player")  # 146
# print(len(extrapolate_all_games([start_game], skill=4)), "games with good player")  # 102 
# print(good_games)
# games = extrapolate_all_games([start_with_center], skill=4)
# games = remove_near_duplicates(games)


def summarize_games(games):
    total_games = len(games)
    first_player_wins = len([x for x in games if check_win_status(x) == 1])
    second_player_wins = len([x for x in games if check_win_status(x) == -1])
    ties = len([x for x in games if check_win_status(x) == 0])
    return total_games, first_player_wins, second_player_wins, ties


def print_games_summary(games):
    total_games, first_player_wins, second_player_wins, ties =  summarize_games(games)
    print(f"{total_games} total games")
    print(f"First player wins {first_player_wins} games ({first_player_wins/total_games:.1%})")
    print(f"Second player wins {second_player_wins} games ({second_player_wins/total_games:.1%})")
    print(f"Tied {ties} games ({ties/total_games:.1%})")


if __name__ == '__main__':

    import pygame
    import numpy as np
    import time

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
                                     (col * square_size + square_size - margin, row * square_size + square_size - margin), int(LINE_WIDTH * 1.2))
                    pygame.draw.line(screen, color, (col * square_size + margin, row * square_size + square_size - margin),
                                     (col * square_size + square_size - margin, row * square_size + margin), int(LINE_WIDTH * 1.2))

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
    #                 inst.play_chord(chord, [0.2, 1.0], frame_dur/3)]
                    s.fork(roll_chord, (inst, chord, 0.7))
                else:
                    inst = ohs if np.sum(delta) == -1 else exes
                    inst.play_note(coords_to_pitch(coords), 0.6, frame_dur/3)
            last_state = game_state
            draw_board(0, 0, WIDTH, HEIGHT)
            draw_xo(game_state, 0, 0, WIDTH, HEIGHT, winning_line, winning_player)
            pygame.display.update()
            time.sleep(frame_dur)


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

    the_20_games = extrapolate_all_games([start_with_center], skill=4)
    the_14_games = remove_near_duplicates(the_20_games)
    game_iter = iter(the_14_games)

    # all_games = extrapolate_all_games([start_game], skill=2)
    # import random
    # random.shuffle(all_games)
    # game_iter = iter(all_games)

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Start the animation (adjust frame_dur as needed)
        animate_game(next(game_iter), frame_dur=0.4)
        time.sleep(1)

    pygame.quit()

    # for game in extrapolate_all_games([start_with_center], skill=4):
    #     print(check_win_status(game))
    #     if check_win_status(game) == -1:
    #         print(game)
    #     print("-----------------------")

    # all_not_dumb_games = extrapolate_all_games([start_game], skill=2)
    # 
    # print(all_not_dumb_games[67])

    # test_game = np.array(
    #     [[[0, 0, 0],
    #       [0, 0, 0],
    #       [0, 0, 0]],
    #      [[0, 0, 0],
    #       [0, 1, 0],
    #       [0, 0, 0]],
    #      [[0, 0, -1],
    #       [0, 1, 0],
    #       [0, 0, 0]],
    #      [[0, 0, -1],
    #       [0, 1, 1],
    #       [0, 0, 0]],
    #      [[0, -1, 0],
    #       [-1, 1, 1],
    #       [0, 0, 0]]]
    # )
    # 
    # print(find_win_setup_locations(test_game[-2], 0))
