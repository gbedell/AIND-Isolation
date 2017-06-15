"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import itertools

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic function is also built around the theory that a corner
    position on the board is a bad position. This function will calculate the
    number of legal moves that result in each player being in a corner of the
    board. Depending on the difference between the opponent's corner moves
    and the given player's corner moves will apply a weight to the final 
    value.

    If the difference results in the opposing player having more possible
    corner moves (a good thing for the given player) a positive weight of 1.5 will
    be applied to the final score.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    corners = [(0,0), (0,game.width), (game.height,0), (game.height,game.width)]

    # Count the number of legal moves that result in the given player
    # being in a corner of the board
    own_moves = game.get_legal_moves(player)
    count_own_corner_moves = 0
    for own_move in own_moves:
        if own_move in corners:
            ++count_own_corner_moves

    # Count the number of legal moves that result in the opponent
    # being in a corner of the board
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    count_opp_corner_moves = 0
    for opp_move in opp_moves:
        if opp_move in corners:
            ++count_opp_corner_moves

    # If the given player has more corner moves than the opponent
    # this is a bad thing, and this will evaluate to a negative value
    net_corner_moves = count_own_corner_moves - count_opp_corner_moves

    # If the given player has more corner moves, apply a negative weight
    if net_corner_moves > 0:
        weight = -1.5 * net_corner_moves
    # If the given player has less corner moves, apply a positive weight
    elif net_corner_moves < 0:
        weight = 1.5 * net_corner_moves
    # If both players have the same number of corner moves, apply
    # (essentially) no weight
    else:
        weight = 1

    # Resulting score is our weight applied to the difference between
    # the given's players moves and the opponent's moves
    return float(weight * (len(own_moves) - len(opp_moves)))

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic function is built around the theory that being in a corner
    of the board puts a player in a bad position. Immediately, if the given
    player is a corner, the minimum possible score (-infinity) is returned. 
    If the given player's opponent position is in the corner, the maximum
    possible score (infinity) is returned. If neither player is in a corner,
    each player's legal moves are evaluated. If either player has a legal move
    that results in the player being in the corner, it is removed from their
    legal moves. The end result is the difference between the given player's
    legal moves minus any corner moves, and the opponent's legal moves minus
    any corner moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    corners = [(0,0), (0,game.width), (game.height,0), (game.height,game.width)]

    # If the given player is in a corner, return the minumum possible value
    if game.get_player_location(player) in corners:
        return float("-inf")

    # If the opponent is in a corner, return the maximum possible value
    if game.get_player_location(game.get_opponent(player)) in corners:
        return float("inf")

    # If neither player is in a corner, continue the evaluation
    
    # Calculate the number of legal moves for a given player minus
    # the player's legal moves that result in a corner position
    own_moves = game.get_legal_moves(player)
    count_own_corner_moves = 0
    for own_move in own_moves:
        if own_move in corners:
            ++count_own_corner_moves
    net_own_moves = len(own_moves) - count_own_corner_moves

    # Calculate the number of legal moves for the opponent minus
    # the opponent's legal moves that result in a corner position
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    count_opp_corner_moves = 0
    for opp_move in opp_moves:
        if opp_move in corners:
            ++count_opp_corner_moves
    net_opp_moves = len(opp_moves) - count_opp_corner_moves

    # Return the difference between the given player's legal moves
    # minus corners and the opponent's legal moves minus corners
    return float(net_own_moves - net_opp_moves)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This heuristic function increases the value for the given player when
    its opponent has potential moves that include a corner. The theory behind
    this, is that being in the corner increases your chance of losing. Thus,
    if the player's opponent has a potential move that includes a corner, we
    want to include the chance of that move happening. The function will
    take the difference between the opponent's legal moves and the given
    player's legal moves, and then add one for every opponent move that
    would result in the opponent being cornered.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    corners = [(0,0), (0,game.width), (game.height,0), (game.height,game.width)]

    opp_moves = game.get_legal_moves(game.get_opponent(player))

    # Count the number of legal moves that result in the opponent
    # ended up in a corner of the board
    count_opp_corner_moves = 0
    for move in opp_moves:
        if move in corners:
            ++count_opp_corner_moves

    count_own_moves = len(game.get_legal_moves(player))
    count_opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # Return the difference in the given player's legal moves and the
    # opponnent's legal moves, plus the number of the opponent's
    # legal moves that result in a corner position
    return float(count_opp_corner_moves + (count_own_moves - count_opp_moves))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def max_value(self, game, depth):
        """Returns the maximum possible value given a game and depth"""
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # Get the legal moves for the given player
        legal_moves = game.get_legal_moves()

        # If there are no legal moves, or we've reached 0 depth, return score
        if not legal_moves or depth == 0:
            return self.score(game, self)

        # Start our return value with the lowest possible maximum
        value = float("-inf")
        for move in legal_moves:
            forecasted_game = game.forecast_move(move)
            value = max(value, self.min_value(forecasted_game, depth-1))
        return value

    def min_value(self, game, depth):
        """Returns the minimum possible value given a game and depth"""
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # Get all legal moves for the given player
        legal_moves = game.get_legal_moves()

        # If there are no legal moves, or we've reached 0 depth, return score
        if not legal_moves or depth == 0:
            return self.score(game, self)

        # Start our return value with the greatest possible minumum
        value = float("inf")
        for move in legal_moves:
            forecasted_game = game.forecast_move(move)
            value = min(value, self.max_value(forecasted_game, depth-1))
        return value

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get all legal moves
        legal_moves = game.get_legal_moves()

        if not legal_moves or depth == 0:
            return game.utility(self)

        # Loop through all legal moves and create a key-value pair
        # in our dictionary of move: score
        moves_dict = dict()
        for move in legal_moves:
            forecasted_game = game.forecast_move(move)
            score = self.min_value(forecasted_game, depth-1)
            moves_dict[move] = score

        # Return the move with the highest score
        return max(moves_dict, key=moves_dict.get)

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.

            # Iterate from 0 to infinity...
            for depth in itertools.count():
                # For each depth we go, continue updating our best_move
                # because we are becoming more intelligent as depth gets
                # deeper
                best_move =  self.alphabeta(game, depth)

        except SearchTimeout:
            # Once we've timed out, return the best move we've found so far
            return best_move

        # Return the best move from the last completed search iteration
        return best_move
    
    def max_value(self, game, depth, alpha, beta):
        """Returns the maximum possible value given a game and depth"""
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get the legal moves for the given player
        legal_moves = game.get_legal_moves()

        # If there are no legal moves, or we've reached 0 depth, return score
        if not legal_moves or depth == 0:
            return self.score(game, self)
        
        # Start our return value with the lowest possible maximum
        value = float("-inf")
        for move in legal_moves:
            forecasted_game = game.forecast_move(move)
            value = max(value, self.min_value(game=forecasted_game, depth=depth-1, alpha=alpha, beta=beta))
            # If value is greater than or equal to the current beta value
            # we can prune the remaining moves
            if value >= beta:
                return value
            # Update alpha with the maximum of value and the minimum score
            alpha = max(alpha, value)
        
        return value

    def min_value(self, game, depth, alpha, beta):
        """Returns the minimum possible value given a game and depth"""
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # Get the legal moves for the given player
        legal_moves = game.get_legal_moves()

        # If there are no legal moves, or we've reached 0 depth, return score
        if not legal_moves or depth == 0:
            return self.score(game, self)
        
        # Start our return value with the greatest possible minimum
        value = float("inf")
        for move in legal_moves:
            forecasted_game = game.forecast_move(move)
            value = min(value, self.max_value(game=forecasted_game, depth=depth-1, alpha=alpha, beta=beta))
            # If value is less than or equal to the current alpha value
            # we can prune the remaining moves
            if value <= alpha:
                return value
            # Update beta with the minimum of value and the maximum score
            beta = min(beta, value)
        
        return value

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get legal moves for given player
        legal_moves = game.get_legal_moves()

        if not legal_moves or depth == 0:
            return game.utility(self)
        
        # if depth == 0:
        #     return self.score(game, self)

        # Loop through all legal moves and create a key-value pair
        # in our dictionary of move: score
        moves_dict = dict()
        for move in legal_moves:
            forecasted_game = game.forecast_move(move)
            score = self.min_value(game=forecasted_game, depth=depth-1, alpha=alpha, beta=beta)
            moves_dict[move] = score
            # Update alpha if the move's score is greater
            alpha = max(alpha, score)
        
        # Return the move with the greatest score
        return max(moves_dict, key=moves_dict.get)  