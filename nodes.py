import numpy as np 
from collections import defaultdict
from abc import ABC, abstractmethod

import time 


# game
from connect_four import Connect4GameState

# print a single row of the board
def stringify( row ):

    return " " + " | ".join(map(lambda x: pieces[int(x)], row)) + " "

# display the whole board
def display( board ):

    board = board.copy().T[::-1]

    for row in board[:-1]:

        print(stringify(row))
        print("-"*(len(row)*4-1))

    print(stringify(board[-1]))
    print()

# monte carlo tree search node
class MonteCarloTreeSearchNode( ABC ):

    def __init__(self, state, parent=None):

        """

        parameters

        state: game  state
        parent: node

        """

        self.state = state
        self.parent = parent
        self.children = []

    @property # use getter and setter
    @abstractmethod # use getter and setter
    def untried_actions(self):
        pass

    @property # usew getter and setter
    @abstractmethod # use getter and setter
    def q(self):
        pass


    @property # use getter and setter
    @abstractmethod # use getter and setter
    def n(self):
        pass

    @abstractmethod # use getter and setter
    def expand(self):
        pass

    @abstractmethod # use getter and setter
    def is_terminal_node(self):
        pass

    @abstractmethod # use getter and setter
    def rollout(self):
        pass

    @abstractmethod # use getter and setter
    def backpropagate(self, reward):
        pass

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):

        choices_weights = [
            (c.q / c.n) + ( c_param * np.sqrt( ( 2 * (np.log(self.n)  / c.n) ) ) ) for c in self.children
        ]

        return self.children[ np.argmax( choices_weights ) ]

    def rollout_policy(self, possible_moves):

        return possible_moves[ np.random.randint( len(possible_moves) ) ]

# two players MCTS
class TwoPlayersGameMonteCarloTreeSearchNode( MonteCarloTreeSearchNode ):

    def __init__(self, state, parent=None):

        super().__init__(state, parent) # make it possible to access a parent class

        self._number_of_visits = 0
        self._results = defaultdict( int )
        self._untried_actions = None

    @property
    def untried_actions(self):

        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()

        return self._untried_actions

    @property
    def q(self):

        wins = self._results[ self.parent.state.next_to_move ]
        loses = self._results[ -1 * self.parent.state.next_to_move ]

        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):

        action = self.untried_actions.pop()
        next_state = self.state.move( action )
        child_node = TwoPlayersGameMonteCarloTreeSearchNode( next_state, parent=self )
        self.children.append( child_node )

        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):

        current_rollout_state = self.state

        while not current_rollout_state.is_game_over():

            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy( possible_moves )
            current_rollout_state = current_rollout_state.move( action )

        return current_rollout_state.game_result

    def backpropagate(self, result):

        self._number_of_visits += 1
        self._results[ result ] += 1

        if self.parent:

            self.parent.backpropagate( result )

class MonteCarloTreeSearch( object ):

    def __init__(self, node):

        self.root = node

    def best_action( self, simulation_number=None, total_simulation_seconds=None ):

        if simulation_number is None:

            assert( total_simulation_seconds is not None )
            end_time = time.time() + total_simulation_seconds

            while True:

                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate( reward )

                if time.time() > end_time:
                    break
        
        else:

            for _ in range( 0, simulation_number ):

                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate( reward )

        return self.root.best_child( c_param=0 )

    def _tree_policy( self ):

        current_node = self.root

        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():

                return current_node.expand()

            else:

                current_node = current_node.best_child()

        return current_node


# define initial state
state = np.zeros( (7, 7) )

# board state
board_state = Connect4GameState( state=state, next_to_move=np.random.choice([-1, 1]), win=4 )

# define pieces
pieces = {
    0: '', 
    1: 'X', 
    -1: 'O'
}

#display( board_state.board )

# play until the game terminates

while board_state.game_result is None:

    # compute best move
    root = TwoPlayersGameMonteCarloTreeSearchNode( state=board_state )
    mcts = MonteCarloTreeSearch( root )
    best_node = mcts.best_action( total_simulation_seconds = 1 )

    # update board 
    board_state = best_node.state

    # display board
    #display( board_state.board )
    print( board_state.board )

print( board_state.game_result, pieces[ board_state.game_result ] )