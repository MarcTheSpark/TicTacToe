import numpy as np
import math
import boardstates
import pickle
from functools import cache
from utils import check_win, find_open_locations, square_format, tuple_format


X_WIN_SCORE = 10
O_WIN_SCORE = -10
DRAW_SCORE = 0


def get_board_score(board):
    if check_win(board, 1):
        return X_WIN_SCORE
    elif check_win(board, -1):
        return O_WIN_SCORE
    elif np.sum(np.abs(board)) == 9:
        return DRAW_SCORE
    else:
        return None
    

def get_move_array(i, j, value):
    """Returns a 2d TTT board with all zeros except for value at coordinates i, j"""
    out = np.zeros((3, 3), dtype=int)
    out[i, j] = value
    return out


@cache
def minimax(current_board: tuple, current_player: int):
    """
    Takes a board as a 9-tuple and a player (1 or -1) and returns the score
    """
    if get_board_score(current_board) is not None:
        return get_board_score(current_board)

    current_board_square = square_format(current_board)
    
    if current_player == 1:
        # x is playing; we want to maximize score
        return max(minimax(tuple_format(current_board_square + get_move_array(i, j, 1)), -current_player)
                   for (i, j) in find_open_locations(current_board_square))
    else:
        # O is playing; we want to maximize score
        return min(minimax(tuple_format(current_board_square + get_move_array(i, j, -1)), -current_player)
                   for (i, j) in find_open_locations(current_board_square))
            
        
def calc_best_moves(board, player):
    board = square_format(board)
    if player == 1:
        best_score = -math.inf
        best_moves = []
        for (i, j) in find_open_locations(board):
            next_board_tuple = tuple_format(board + get_move_array(i, j, player))
            board_score = minimax(next_board_tuple, -player)
            if board_score == best_score:
                best_moves.append(next_board_tuple)
            elif board_score > best_score:
                best_moves = [next_board_tuple]
                best_score = board_score
    else:  # player == -1
        best_score = math.inf
        best_moves = []
        for (i, j) in find_open_locations(board):
            next_board_tuple = tuple_format(board + get_move_array(i, j, player))
            board_score = minimax(next_board_tuple, -player)
            if board_score == best_score:
                best_moves.append(next_board_tuple)
            elif board_score < best_score:
                best_moves = [next_board_tuple]
                best_score = board_score

    return best_moves


def get_all_rotations_and_reflections(square_board):
    return [
        np.flip(np.rot90(square_board, rot, axes=(0, 1)), 1) if flip
        else np.rot90(square_board, rot, axes=(0, 1))
        for rot in range(4)
        for flip in range(2)
    ]


if __name__ == '__main__':
    best_moves_dict = {}
    for i, states_after_move_n in enumerate(boardstates.BoardStatesList.complete().board_states):
        for state in tuple_format(states_after_move_n):
            best_moves_dict[state] = calc_best_moves(state, 1 if i % 2 == 0 else -1)
    with open('minimax.pickle', 'wb') as handle:
        pickle.dump(best_moves_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    with open('minimax.pickle', 'rb') as handle:
        best_moves_dict = pickle.load(handle)


def get_best_moves(board):
    board = tuple_format(board)
    next_moves = [np.array(next_board) for next_board in best_moves_dict[board]]
    next_move_indices = [int(np.where(next_move != board)[0]) for next_move in next_moves]
    return [divmod(x, 3) for x in next_move_indices]


if __name__ == '__main__':
    print(get_best_moves((0, 1, 0, -1, 0, 0, 0, 0, 0)))