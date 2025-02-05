import matplotlib.pyplot as plt
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

#Global dictionary to store state information
#Key: (board, player) where board is a tuple of 9 characters and player is 'X' or 'O'
#Value: a dict with keys 'terminal', 'reward' and 'actions'. The 'actions' key
#       maps moves (cell index) to the resulting state

state_info = {}

def generate_states(board, player):
    """
    Recursively generates all reachable states from the given board and player.

    Args:
        board: A tuple of 9 characters representing the board
        player: a string, either 'X' or 'O' indicating whose turn it is
    """

    state = (board, player)
    if state in state_info:
        return 
    terminal, reward = is_terminal(board)
    state_info[state] = {'terminal': terminal, 'reward': reward, 'actions': {}}

    moves = get_available_moves(board)
    for move in moves:
        new_board = apply_move(board, move, player)
        next_player = 'O' if player == 'X' else 'X'
        next_state = (new_board, next_player)
        state_info[state]['actions'][move] = next_state
        #recursively get states from successor state
        generate_states(new_board, next_player)

def value_iteration(state_info, gamma=1.0, epsilon = 1e-5):
    """
    Performs value iteration over the state space, recording the delta at each iteration

    Args:
        state_info: the dictionary of state information
        gamme: discount factor
        epsilon: convergence threshold

    Returns:
        a tuple (V, iterations, deltas) 
        where V is a dictionary mapping states to their computed value, 
        iterations is the number of iterations required,
        deltas is a list of the maximum changes in value per iteration
    """

    V = {}
    deltas = []
    for state, info in state_info.items():
        if info['terminal']:
            V[state] = info['reward']
        else:
            V[state] = 0.0

    iterations = 0
    while True:
        delta = 0.0
        newV = {}
        for state, info in state_info.items():
            if info['terminal']:
                newV[state] = info['reward']
            else:
                board, player = state
                if player == 'X':
                    best_val = -float('inf')
                    for move, next_state in info['actions'].items():
                        next_info = state_info[next_state]
                        #use the reward if terminal; otherwise, the current value
                        val = next_info['reward'] if next_info['terminal'] else V[next_state]
                        best_val = max(best_val, val)
                    newV[state] = best_val
                else: #O's turn: average
                    total = 0.0
                    count = 0
                    for move, next_state in info['actions'].items():
                        next_info = state_info[next_state]
                        val = next_info['reward'] if next_info['terminal'] else V[next_state]
                        total += val
                        count += 1
                    newV[state] = total/count if count >0 else 0.0
            delta = max(delta, abs(newV[state] - V[state]))
        V = newV
        iterations +=1
        deltas.append(delta) #append the current delta to the list
        if delta < epsilon:
            break
    return V, iterations, deltas

if __name__ == '__main__':
    #start from an empty board with player 'x' to move
    initial_board = tuple(' ' * 9)
    generate_states(initial_board, 'X')
    print("Number of states generated:", len(state_info))

    #run value iteration over the generated state pace
    V, num_iterations, deltas = value_iteration(state_info)
    print("Value iteration converged in:", num_iterations, "iterations")

    initial_state = (initial_board, 'X')
    print("Value of the initial state:", V[initial_state])

    plt.figure(figsize=(10, 6))
    plt.plot(deltas, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Max Delta')
    plt.title('Convergence of Value Iteration')
    plt.grid(True)
    plt.show()





