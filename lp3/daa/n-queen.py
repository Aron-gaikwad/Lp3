def print_solution(board):
    for row in board:
        print(" ".join("Q" if col else "." for col in row))
    print()


def is_safe(board, row, col):
    # Check this column on upper side
    for i in range(row):
        if board[i][col]:
            return False

    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if j < 0:  # Check column bounds
            break
        if board[i][j]:
            return False

    # Check upper diagonal on right side
    for i, j in zip(range(row, -1, -1), range(col, len(board))):
        if j >= len(board):  # Check column bounds
            break
        if board[i][j]:
            return False

    return True


def solve_n_queens_util(board, row):
    if row >= len(board):
        print_solution(board)
        return

    for col in range(len(board)):
        if is_safe(board, row, col):
            board[row][col] = True  # Place queen
            solve_n_queens_util(board, row + 1)  # Recur to place rest of the queens
            board[row][col] = False  # Backtrack


def solve_n_queens(n):
    board = [[False] * n for _ in range(n)]  # Create an n x n board
    solve_n_queens_util(board, 0)


# Main program
if __name__ == "__main__":
    n = int(input("Enter the number of queens: "))
    solve_n_queens(n)
