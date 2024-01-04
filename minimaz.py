import numpy as np
import math
import boardstates
import pickle

X_WIN_SCORE = 10
O_WIN_SCORE = -10
DRAW_SCORE = 0


def check_win(board, player):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if all(board[i, :] == player) or all(board[:, i] == player):
            return True
    if all(np.diag(board) == player) or all(np.diag(np.fliplr(board)) == player):
        return True
    return False


def get_board_score(board):
    if check_win(board, 1):
        return X_WIN_SCORE
    elif check_win(board, -1):
        return O_WIN_SCORE
    elif np.sum(np.abs(board)) == 9:
        return DRAW_SCORE
    else:
        return None
    

def find_open_locations(board):
    """Just get all the zeros"""
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

def get_move_array(i, j, value):
    out = np.zeros((3, 3))
    out[i, j] = value
    return out

def minimax(current_board, current_player):
    if get_board_score(current_board) is not None:
        return get_board_score(current_board)
    
    if current_player == 1:
        # x is playing; we want to maximize score
        return max(minimax(current_board + get_move_array(i, j, 1), -current_player)
                   for (i, j) in find_open_locations(current_board))
    else:
        # O is playing; we want to maximize score
        return min(minimax(current_board + get_move_array(i, j, -1), -current_player)
                   for (i, j) in find_open_locations(current_board))
            
        
def calc_best_moves(board, player):
    if player == 1:
        best_score = -math.inf
        best_moves = []
        for (i, j) in find_open_locations(board):
            next_board = board + get_move_array(i, j, player)
            board_score = minimax(next_board, -player)
            if board_score == best_score:
                best_moves.append(next_board)
            elif board_score > best_score:
                best_moves = [next_board]
                best_score = board_score
    else:  # player == -1
        best_score = math.inf
        best_moves = []
        for (i, j) in find_open_locations(board):
            next_board = board + get_move_array(i, j, player)
            board_score = minimax(next_board, -player)
            if board_score == best_score:
                best_moves.append(next_board)
            elif board_score < best_score:
                best_moves = [next_board]
                best_score = board_score
        
        
    return best_moves


def get_all_rotations_and_reflections(flattened_board_state):
    regular_board_state = flattened_board_state.reshape((3, 3))
    return [
        np.flip(np.rot90(regular_board_state, rot, axes=(0, 1)), 1).flatten() if flip
            else np.rot90(regular_board_state, rot, axes=(0, 1)).flatten()
        for rot in range(4)
        for flip in range(2)
    ]


if False: #__name__ == '__main__':
    best_moves_dict = {}

    for i, states_after_move_n in enumerate(boardstates.all_board_states):
        for state in states_after_move_n:
            best_moves_dict[tuple(state.flatten())] = [
                tuple(x.flatten())
                for x in calc_best_moves(state.reshape((3, 3)), 1 if i % 2 == 0 else -1)
            ]

    with open('minimax.pickle', 'wb') as handle:
        pickle.dump(best_moves_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
else:
    with open('minimax.pickle', 'rb') as handle:
        best_moves_dict = pickle.load(handle)
            
            
def get_best_moves(board):
    board = tuple(np.array(board).flatten())
    next_moves = [np.array(next_board) for next_board in best_moves_dict[board]]
    next_move_indices = [int(np.where(next_move != board)[0]) for next_move in next_moves]
    return [divmod(x, 3) for x in next_move_indices]

