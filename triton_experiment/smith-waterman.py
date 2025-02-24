import numpy as np

# Define scoring parameters
MATCH_SCORE = 3
MISMATCH_PENALTY = -3
GAP_PENALTY = -2

def smith_waterman(X, Y):
    # Initialize the scoring matrix
    m, n = len(X), len(Y)
    H = np.zeros((m + 1, n + 1), dtype=int)  # H[i][j] will store the score for X[0:i] vs Y[0:j]

    # Fill the scoring matrix H
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = H[i-1][j-1] + (MATCH_SCORE if X[i-1] == Y[j-1] else MISMATCH_PENALTY)
            delete = H[i-1][j] + GAP_PENALTY  # gap in Y (deletion)
            insert = H[i][j-1] + GAP_PENALTY  # gap in X (insertion)
            H[i][j] = max(0, match, delete, insert)  # Smith-Waterman chooses the maximum score or 0

    # Find the highest score in the matrix
    max_score = np.max(H)
    max_pos = np.unravel_index(np.argmax(H), H.shape)  # Find the position of the highest score

    # Traceback to find the optimal local alignment
    aligned_X, aligned_Y = [], []
    i, j = max_pos

    while H[i][j] > 0:  # Traceback stops when score is 0
        current_score = H[i][j]
        diagonal_score = H[i-1][j-1] + (MATCH_SCORE if X[i-1] == Y[j-1] else MISMATCH_PENALTY)
        up_score = H[i-1][j] + GAP_PENALTY
        left_score = H[i][j-1] + GAP_PENALTY

        # Decide direction of traceback
        if current_score == diagonal_score:
            aligned_X.append(X[i-1])
            aligned_Y.append(Y[j-1])
            i -= 1
            j -= 1
        elif current_score == up_score:
            aligned_X.append(X[i-1])
            aligned_Y.append('-')  # Gap in Y
            i -= 1
        elif current_score == left_score:
            aligned_X.append('-')  # Gap in X
            aligned_Y.append(Y[j-1])
            j -= 1

    # Reverse the alignments to get the correct order
    aligned_X = ''.join(reversed(aligned_X))
    aligned_Y = ''.join(reversed(aligned_Y))

    return max_score, aligned_X, aligned_Y

# Test the Smith-Waterman function with an example
X = "TGTTACGG"
Y = "GGTTGACTACGAACG"

score, aligned_X, aligned_Y = smith_waterman(X, Y)
print(f"Optimal Score: {score}")
print(f"Aligned X: {aligned_X}")
print(f"Aligned Y: {aligned_Y}")
