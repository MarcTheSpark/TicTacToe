import numpy as np


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
    """
    extrapolated = []
    # Determine whose turn it is (1 for O's turn, -1 for X's turn)
    current_board = game[-1]
    whose_turn = 1 if np.sum(current_board) % 2 == 0 else -1
    
    winning_moves = find_winning_locations(current_board)
    player_wins, opponent_wins = winning_moves if whose_turn == 1 else winning_moves[::-1]
    
    if skill > 0 and len(player_wins) > 0:
        next_moves = player_wins
    elif skill > 1 and len(opponent_wins) > 0:
        next_moves = opponent_wins
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


def prune_dumb_moves(games):
    """Looks through a list of games and removes those where the last move is dumb."""
    winning_moves = [game for game in games if check_win_status(game) in (-1, 1)]
    if len(winning_moves) == 0:
        return games
    else:
        return winning_moves
    

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


def extrapolate_all_games(unfinished_games, skill=0):
    
    finished_games = []
    
    for game in reversed(unfinished_games):
        if is_game_over(game):
            unfinished_games.remove(game)
            finished_games.append(game)
                
    while len(unfinished_games) > 0:
        print(f"PRE: {len(unfinished_games)}, {len(finished_games)}")
        unfinished_games = [
            extrapolated_game
            for game in unfinished_games
            for extrapolated_game in prune_symmetrical_games(extrapolate_one_step(game, skill=skill))
        ]
        print(f"EXTRAPOLATE: {len(unfinished_games)}, {len(finished_games)}")
        for i in reversed(range(len(unfinished_games))):
            if is_game_over(unfinished_games[i]):
                finished_games.append(unfinished_games[i])
                del unfinished_games[i]
        print(f"PRUNE ENDED: {len(unfinished_games)}, {len(finished_games)}")
#         if len(unfinished_games) < 400:
#             prune_symmetrical_games(unfinished_games)
#             print(f"PRUNE SYM: {len(unfinished_games)}, {len(finished_games)}")
    return finished_games
       
       
start_game = np.array(
    [[[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]]]
)

# print(extrapolate_one_step(start_game))
# exit()

# print(len(extrapolate_all_games([start_game])), "games in total")  # 31896
# print(len(extrapolate_all_games([start_game], skill=1)), "games with idiotic moves removed")  # 6956
# print(len(extrapolate_all_games([start_game], skill=2)), "games with dumb moves removed")  # 2936

# all_not_dumb_games = extrapolate_all_games([start_game], skill=2)
# 
# print(all_not_dumb_games[67])

# next level: play in a spot that gives you 2 in a row with open 3rd where possible

# 31896 games after symmetry

# for i, extrapolated_game in enumerate(extrapolated):
#     print(f"GAME {i+1}")
#     print(extrapolated_game)
            
# extrapolated = extrapolate_one_step(all_games[0])
# prune_symmetrical_games(extrapolated)
# print(extrapolated)

game1 = np.array(
    [[[0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]],
     [[0, 0, 0],
      [0, 1, 0],
      [0, 0, 0]],
     [[0, 0, -1],
      [0, 1, 0],
      [0, 0, 0]],
     [[0, 0, -1],
      [0, 1, 1],
      [0, 0, 0]],
     [[0, -1, 0],
      [-1, 1, 1],
      [0, 0, 0]]]
)

# game2 = np.array(
#     [[[1, 0, 0],
#       [0, -1, 1],
#       [0, 0, 1]],
#      [[0, 0, 0],
#       [0, 0, 0],
#       [1, 1, -1]]]
# )

print(is_game_over(game1[:-1]))
# print(game1)

# print(are_symmetrical(np.array(game1), np.array(game2)))