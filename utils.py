import numpy as np


# ------------------ Formatting Utils ------------------------


def flat_format(game_or_board: np.ndarray):
    if not isinstance(game_or_board, np.ndarray):
        game_or_board = np.array(game_or_board)
    if game_or_board.shape[-1:] == (9, ):
        return game_or_board
    else:
        return game_or_board.reshape(game_or_board.shape[:-2] + (9,))


def square_format(game_or_board: np.ndarray):
    if not isinstance(game_or_board, np.ndarray):
        game_or_board = np.array(game_or_board)
    if game_or_board.shape[-2:] == (3, 3):
        return game_or_board
    else:
        return game_or_board.reshape(game_or_board.shape[:-1] + (3, 3))


def ndarray_to_tuple(arr):
    if arr.ndim == 1:
        return tuple(arr)
    return tuple(ndarray_to_tuple(arr[i]) for i in range(len(arr)))


def tuple_format(game_or_board):
    return ndarray_to_tuple(flat_format(game_or_board))


# -------------------------- board utils ----------------------------


def check_win(board, player):
    board = square_format(board)
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if all(board[i, :] == player) or all(board[:, i] == player):
            return True
    if all(np.diag(board) == player) or all(np.diag(np.fliplr(board)) == player):
        return True
    return False


def find_open_locations(board):
    """Find coordinates of all the zeros for a square formatted board"""
    board = square_format(board)
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]


def check_win(board, player):
    board = square_format(board)
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if all(board[i, :] == player) or all(board[:, i] == player):
            return True
    if all(np.diag(board) == player) or all(np.diag(np.fliplr(board)) == player):
        return True
    return False


def find_winning_locations(board):
    """
    Identify all O-win-locations and X-win-locations on a Tic-Tac-Toe board.

    :param board: A 3x3 NumPy array representing the Tic-Tac-Toe board.
    :return: Two lists containing the coordinates of winning locations for O and X.
    """
    board = square_format(board)

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


# ---------------------------------- Game utils ------------------------------------------


def are_symmetrical(game1, game2):
    """
    Check if two Tic-Tac-Toe games are rotationally or mirror symmetrical.

    :param game1: A 3D NumPy array representing the first Tic-Tac-Toe game (nx3x3).
    :param game2: A 3D NumPy array representing the second Tic-Tac-Toe game (nx3x3).
    :return: True if the games are symmetrical, False otherwise.
    """
    game1 = square_format(game1)
    game2 = square_format(game2)

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
    game = square_format(game)

    # Start with an empty board
    if not np.array_equal(game[0], np.zeros((3, 3))):
        return False

    for i in range(1, game.shape[0]):
        # Check the sum for odd and even turns
        if np.sum(game[i]) != i % 2:
            return False

        # Check that there is exactly one difference from the previous turn
        if np.sum(np.abs(game[i] - game[i - 1])) != 1:
            return False

    return True


def check_win_status(game):
    """
    Check if a Tic-Tac-Toe game is over and if so, who won

    :param game: A 3D NumPy array representing a Tic-Tac-Toe game (nx3x3).
    :return: -1 if X's win, 1 if O's win, and 0 if draw; False if not over
    """
    game = square_format(game)

    last_board = game[-1]
    if check_win(last_board, 1):
        return 1  # O wins
    elif check_win(last_board, -1):
        return -1  # X wins
    elif np.sum(np.abs(last_board)) == 9:
        return 0  # draw
    else:
        return None  # unfinished


def is_game_over(game):
    """
    Check if a Tic-Tac-Toe game is over by examining the last frame.

    :param game: A 3D NumPy array representing a Tic-Tac-Toe game (nx3x3).
    :return: True if the game is over (win or draw), False otherwise.
    """
    game = square_format(game)
    return check_win_status(game) is not None


def _find_differing_indices(tuple1, tuple2):
    return [i for i, (x, y) in enumerate(zip(tuple1, tuple2)) if x != y]


def get_move_sequence(game):
    game = tuple_format(game)
    return tuple(_find_differing_indices(game[i], game[i+1])[0] for i in range(len(game) - 1))

# ---------------------------------------- stats ----------------------------------------------


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