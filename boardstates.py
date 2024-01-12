from itertools import combinations
import math
import numpy as np
from utils import square_format, tuple_format


def get_all_rotations_and_reflections(board_state: tuple):
    board_state = square_format(board_state)
    return set(
        tuple_format(np.flip(np.rot90(board_state, rot, axes=(0, 1)), 1)) if flip
        else tuple_format(np.rot90(board_state, rot, axes=(0, 1)))
        for rot in range(4)
        for flip in range(2)
    )


def get_standard_form(board_state):
    return min(get_all_rotations_and_reflections(board_state))


def get_all_board_states_after_move_n(n):
    """
    Returns a list of board states after the nth move as a 9-tuple
    """
    board_states = []
    num_x_moves = math.ceil(n / 2)
    num_o_moves = n - num_x_moves
    for x_moves in combinations(range(1, 10), num_x_moves):
        remaining_moves = tuple(x for x in range(1, 10) if x not in x_moves)
        for o_moves in combinations(remaining_moves, num_o_moves):
            board_states.append(tuple(1 if i in x_moves else -1 if i in o_moves else 0 for i in range(1, 10)))
    return board_states


class BoardStatesList:

    def __init__(self, board_states_after_each_move):
        self.board_states = [[tuple_format(board) for board in sorted(x)] for x in board_states_after_each_move]
        while len(self.board_states) < 10:
            self.board_states.append([])

    @classmethod
    def from_games(cls, games):
        """
        Get a list of all board states used in the given games.
        """
        used_states = [set() for _ in range(10)]
        for game in games:
            for i, state in enumerate(game):
                used_states[i].add(state)
        return cls(used_states)

    def standard_forms_only(self):
        """
        Returns a copy of the same board states list pruned to only contain the standard rotation/reflection
        of each board state.
        """
        return BoardStatesList([
            tuple(sorted(set(
                get_standard_form(board_state)
                for board_state in board_states_after_move_n
            )))
            for board_states_after_move_n in self.board_states
        ])

    def get_state_index(self, board_state):
        """Given a board state as a 9-tuple, returns its move number and index."""
        move_num = sum(abs(x) for x in board_state)
        this_move_states = self.board_states[move_num]
        try:
            return move_num, this_move_states.index(board_state)
        except ValueError:
            return None

    def index_sequence_to_board_sequence(self, state_sequence):
        return [self.board_states[i][state_num] for i, state_num in enumerate(state_sequence)]

    @classmethod
    def complete(cls):
        return cls([get_all_board_states_after_move_n(n) for n in range(10)])

    def sizes_per_move(self):
        return tuple(len(x) for x in self.board_states)

    def __repr__(self):
        return f"BoardStatesIndex({self.board_states})"


if __name__ == '__main__':
    complete_index = BoardStatesList.complete()
    print(complete_index)
    print(complete_index.sizes_per_move())
    print(complete_index.standard_forms_only().sizes_per_move())
    print(complete_index.get_state_index((0, 0, 1, 0, -1, 0, 0, 0, 0)))
    print(get_standard_form((0, 0, 1, 0, -1, 0, 0, 0, 0)))
