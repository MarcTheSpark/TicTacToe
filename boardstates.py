from itertools import combinations
import math
import numpy as np


def get_all_board_states_after_move_n(n):
    board_states = []
    num_x_moves = math.ceil(n/2)
    num_o_moves = n - num_x_moves
    for x_moves in combinations(range(1, 10), num_x_moves):
        remaining_moves = tuple(x for x in range(1, 10) if x not in x_moves)
        for o_moves in combinations(remaining_moves, num_o_moves):
            board_states.append(np.array([
                1 if i in x_moves else -1 if i in o_moves else 0 for i in range(1, 10)]))
    return board_states


all_board_states = [np.array(get_all_board_states_after_move_n(n)) for n in range(10)]

if __name__ == '__main__':
    print(all_board_states[0])
    print([len(x) for x in all_board_states])
