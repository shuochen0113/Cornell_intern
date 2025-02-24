import triton
import triton.language as tl

import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Scoring scheme
MATCH = 3
MISMATCH = -3
GAP = -2

@triton.jit
def smith_waterman_kernel(
    # Pointers for input sequences
    seq_m_ptr, seq_n_ptr, 
    # Pointer for the DP matrix
    matrix_ptr, 
    # Dimensions of the DP matrix
    m, n, 
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr):

    # There are multiple 'programs' processing different data. We identify which program.
    # Try to do them in "small groups" in the matrix.
    pid = tl.program_id(axis=0)


    # Determine the pointer for the sequences this pid is responsible for
    'Check the correctness of this part'

    current_idx = pid + 1
    
    if current_idx <= (n*(n+1))//2:
        upper_bound_idx = 1
        while (upper_bound_idx * (upper_bound_idx + 1)) // 2 < current_idx:
            upper_bound_idx += 1
        upper_bound = (upper_bound_idx * (upper_bound_idx + 1)) // 2
        lower_bound = ((upper_bound_idx - 1) * upper_bound_idx) // 2

        num_diagnol = upper_bound - lower_bound
        diff = upper_bound - current_idx

        m_start = (num_diagnol - diff - 1) * BLOCK_SIZE
        n_start = (diff) * BLOCK_SIZE

    elif m != n:
        sub_mn = m - n
        flag1 = (n*(n+1))//2+n*sub_mn
        if current_idx <= flag1:
            hori_move = 0
            upper_bound = (n*(n+1))//2
            while upper_bound < current_idx:
                upper_bound += n
                hori_move += 1

            diff = upper_bound - current_idx
            m_start = (n + hori_move - diff - 1) * BLOCK_SIZE
            n_start = (diff) * BLOCK_SIZE

        else:
            upper_bound = (n*(n+1))//2+n*sub_mn
            nn = n-1
            while upper_bound < current_idx:
                upper_bound += nn
                nn -= 1
            
            hori_move = n-nn
            diff = upper_bound - current_idx
            m_start = (n + sub_mn - diff - 1) * BLOCK_SIZE
            n_start = (diff + hori_move) * BLOCK_SIZE

    else:
        upper_bound = (n*(n+1))//2
        nn = n-1
        while upper_bound < current_idx:
            upper_bound += nn
            nn -= 1

        hori_move = n-nn
        diff = upper_bound - current_idx
        m_start = (n - diff - 1) * BLOCK_SIZE
        n_start = (diff + hori_move) * BLOCK_SIZE

    matrix_m_start = m_start - 1
    matrix_n_start = n_start - 1

    # Load sequences and matrix
    off_m = m_start + tl.arange(0, BLOCK_SIZE)
    off_n = n_start + tl.arange(0, BLOCK_SIZE)
    mask_m = off_m < m
    mask_n = off_n < n
    seq_m = tl.load(seq_m_ptr + off_m, mask=mask_m)
    seq_n = tl.load(seq_n_ptr + off_n, mask=mask_n)

    off_matrix_m = matrix_m_start + tl.arange(0, BLOCK_SIZE+1)
    off_matrix_n = matrix_n_start + tl.arange(0, BLOCK_SIZE+1)
    mask_matrix_m = off_matrix_m < m
    mask_matrix_n = off_matrix_n < n
    matrix_ptrs = matrix_ptr + (off_matrix_m[:, None] + off_matrix_n[None, :])
    matrix_mask = (mask_matrix_m[:, None] & mask_matrix_n[None, :])
    matrix = tl.load(matrix_ptrs, mask=matrix_mask)

    # Do the smith-waterman computation
    for i in range(1, len(seq_m)+1):
        for j in range(1, len(seq_n)+1):
            diagonal_score = matrix[i-1, j-1] + (MATCH if seq_m[i-1] == seq_n[j-1] else MISMATCH)
            up_score = matrix[i-1, j] + GAP  # gap in Y (deletion)
            left_score = matrix[i, j-1] + GAP  # gap in X (insertion)

    tl.store(matrix_ptrs, matrix, mask=matrix_mask)

def smith_waterman(seq_1, seq_2):
    BLOCK_SIZE = 16 # adjustable
    len1 = len(seq_1)
    len2 = len(seq_2)

    if len1 >= len2:
        seq_m = seq_1
        seq_n = seq_2
        m = len1
        n = len2
    else:
        seq_m = seq_2
        seq_n = seq_1
        m = len2
        n = len1

    # Determine the total number of "parallel computation" loops
    num_pid_m = m // BLOCK_SIZE + 1
    num_pid_n = n // BLOCK_SIZE + 1
    big_pid = max(num_pid_m, num_pid_n)
    small_pid = min(num_pid_m, num_pid_n)
    diff_pid = big_pid - small_pid

    # num_pid = num_pid_m * num_pid_n
    # k = 1
    # while (k * (k + 1)) // 2 < num_pid:
    #     k += 1

    # Initialize the resulat DP matrix
    matrix = torch.zeros((m+1), (n+1))
    assert seq_m.device == DEVICE and seq_n.device == DEVICE and matrix.device == DEVICE

    # Launch the Triton kernel in a specific order to avoid data dependency problem
    for i in range (big_pid):
        if i + 1 <= small_pid:
            grid_num = (i+1) * (i+2) // 2
        elif diff_pid == 0:
            j = 2 * small_pid - (i + 1)
            grid_num = (j * (j + 1)) // 2
        else:
            if i + 1 - small_pid <= diff_pid:
                grid_num = (small_pid * (small_pid + 1)) // 2
            else:
                j = 2 * small_pid - (i + 1) + diff_pid
                grid_num = (j * (j + 1)) // 2
        grid = lambda meta: (grid_num * meta['BLOCK_SIZE'], meta['BLOCK_SIZE'])
        smith_waterman_kernel[grid](seq_m, seq_n, matrix, m, n, BLOCK_SIZE)

    return matrix


# Example usage:
seq_m = [1, 2, 3, 4, 5]  # Example: encoded sequence (could represent nucleotides)
seq_n = [1, 2, 3]

m, n = len(seq_m), len(seq_n)
result_matrix = smith_waterman(seq_m, seq_n)
print(result_matrix)