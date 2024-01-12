from boardstates import BoardStatesList
from minimax import get_best_moves
from game_extrapolation import find_winning_locations, find_win_setup_locations, find_open_locations
import numpy as np
from utils import square_format


def get_heuristic_moves(current_board, skill=0):
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

    whose_turn = 1 if np.sum(current_board) % 2 == 0 else -1

    winning_moves = find_winning_locations(current_board)
    player_wins, opponent_wins = winning_moves if whose_turn == 1 else winning_moves[::-1]

    if skill >= 1 and len(player_wins) > 0:
        return  player_wins
    elif skill >= 2 and len(opponent_wins) > 0:
        return  opponent_wins
    elif skill >= 3:
        # using more "sophisticated" strategies
        if skill >= 7:
            # Perfect (minimax) skill (what does this mean exactly)
            return get_best_moves(current_board)
        elif skill >= 6:
            # Minimax
            minimax_next_moves =  get_best_moves(current_board)
            # Heuristic (fork if we can)
            good_plays = find_win_setup_locations(current_board, whose_turn, only_the_best=True)
            # mix of minimax and heuristic
            return sorted(set(good_plays + minimax_next_moves))
        elif skill >= 5:
            # Minimax
            minimax_next_moves = get_best_moves(current_board)
            # Heuristic (fork if we can)
            good_plays = find_win_setup_locations(current_board, whose_turn, only_the_best=False)
            # mix of minimax and heuristic
            return sorted(set(good_plays + minimax_next_moves))
        else:
            good_plays = find_win_setup_locations(current_board, whose_turn, only_the_best=skill >= 4)
            if len(good_plays) > 0:
                return  good_plays
            else:
                return find_open_locations(current_board)
    else:
        return find_open_locations(current_board)


for state in BoardStatesList.complete().board_states[3]:
    heuristic_moves = set(get_heuristic_moves(square_format(state), 4))
    best_moves = set(get_best_moves(state))
    num_open_moves = len(find_open_locations(state))
    # union should not be everything, neither should be a subset of the other
    if best_moves.issubset(heuristic_moves) or heuristic_moves.issubset(best_moves) or len(heuristic_moves.union(best_moves)) == num_open_moves:
        continue
    print(state)
    print(heuristic_moves)
    print(best_moves)
    print("---")
