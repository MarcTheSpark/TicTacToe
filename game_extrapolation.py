"""There are exactly 20 different* games of Tic-Tac-Toe"""

import numpy as np
from minimax import get_best_moves
from utils import (find_open_locations, is_game_over, find_winning_locations, find_win_setup_locations, are_symmetrical,
                   check_win_status, summarize_games, print_games_summary)
from specific_games import start_game, start_with_center


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


if __name__ == '__main__':
    # 9! = 362880, but this doesn't account for games that end early
    print(len(extrapolate_all_games([start_game], prune_symmetrical=False)), "games in total")  # 255168
    print(len(extrapolate_all_games([start_game])), "games in total")  # 31896
    print(len(extrapolate_all_games([start_game], skill=1)), "games with idiotic moves removed")  # 6956
    print(len(extrapolate_all_games([start_game], skill=2)), "games with dumb moves removed")  # 2936
    print(len(extrapolate_all_games([start_game], skill=3)), "games with decent player")  # 146
    print(len(extrapolate_all_games([start_game], skill=4)), "games with good player")  # 102
    games = extrapolate_all_games([start_with_center], skill=4)
    games = remove_near_duplicates(games)
