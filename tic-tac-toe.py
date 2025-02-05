"""
Tic Tac Toe Value Iteration

Generates all reachable tic tac toe states (from the perspective of playe 'X')
then uses value iteration to compute the value for each state. The reward structure is:
- X wins: +1
- O wins: -1
- Draw: 0
For nonterminal states: reward is 0

Player X (the agent) is assumed to act optimally
while player 0 is assumed to act randomly (averaging over available moves)

When you run the script it will print:
- The number of states generated
- The number of iterations needed for value iteration to converge
- The value of the initial state
"""

#def print_board(board):
 #   """Prints the board in 3x3 format"""
  #  for i in range(3):
   #     print(board[3*i:3*i+3])
    #print()

#print_board(3)

def print_board(size):
    board = [i for i in range(size * size)]  # Example initialization of the board
    for i in range(size):
        print(board[size * i:size * i + size])



def check_winner(board):
    """
    Checks if there's a winner on the board

    Returns:
    'X' if player X wins,
    'O' if player O wins,
    None otherwise
    """

    #winning combinations can be rows, columns or diagonals
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
             (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6)]
    for (i, j, k) in lines:
        if board[i] != ' ' and board[i]==board[j]==board[k]:
            return board[i]
    return None


# Example board for testing
board = ['X', 'O', 'O', 'X', 'O', ' O', 'X', ' ', ' ']
print_board(3)
winner = check_winner(board)
print(f"Winner: {winner}")

def is_terminal(board):
    """
    Determines if the board is in a terminal state

    Returns:
    (terminal_flag, reward) where reward is from X's perspective:
    +1 if X wins, -1 if 0 wins, 0 for a draw or nonterminal state
    """
    winner = check_winner(board)
    if winner is not None:
        reward = 1 if winner == 'X' else -1
        return True, reward
    elif ' ' not in board:
        #board is full and no winner
        return True, 0
    else:
        return False, 0
    
terminal = is_terminal(board)
print(f"Game ended, reward: {terminal}")

def get_available_moves(board):
    """
    Returns a list of indices (0-8) corresponding to empty cells
    """
    return [i for i, cell in enumerate(board) if cell == ' ']

available_moves = get_available_moves(board)
print(f"Available cells: {available_moves}")

def apply_move(board, move, player):
    """
    Returns a new board after the player makes a move at the given index
    """
    board = list(board)
    board[move] = player
    return tuple(board)

move = available_moves[0]  # Example move
player = 'O'  # Example player

new_board = apply_move(board, move, player)
print(f"New board: {new_board}")
