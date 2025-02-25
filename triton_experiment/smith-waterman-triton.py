import time
import random
import torch
import triton
import triton.language as tl

# ------------------------- CPU 版本 -------------------------
def smith_waterman_cpu(seq1, seq2, match=2, mismatch=-1, gap=-1):
    m, n = len(seq1), len(seq2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    max_score = 0
    max_pos = (0, 0)
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            match_score = dp[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            delete = dp[i-1][j] + gap
            insert = dp[i][j-1] + gap
            dp[i][j] = max(match_score, delete, insert, 0)
            
            if dp[i][j] > max_score:
                max_score = dp[i][j]
                max_pos = (i, j)
    
    return max_score, max_pos, dp

# ------------------------- GPU 版本 -------------------------
@triton.jit
def sw_kernel(seq1_ptr, seq2_ptr, dp_ptr, k,
              m, n, match, mismatch, gap,
              seq1_stride, seq2_stride, dp_stride,
              BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    i_min = max(1, k - n)
    i_max = min(m, k - 1)
    num_valid = i_max - i_min + 1
    
    mask = idx < num_valid
    i = i_min + idx
    j = k - i
    
    valid = (i >= 1) & (i <= m) & (j >= 1) & (j <= n)
    mask &= valid
    
    c1 = tl.load(seq1_ptr + (i-1)*seq1_stride, mask=mask, other=0)
    c2 = tl.load(seq2_ptr + (j-1)*seq2_stride, mask=mask, other=0)
    
    diag = tl.load(dp_ptr + (i-1)*dp_stride + (j-1), mask=mask, other=0)
    up   = tl.load(dp_ptr + (i-1)*dp_stride + j, mask=mask, other=0)
    left = tl.load(dp_ptr + i*dp_stride + (j-1), mask=mask, other=0)
    
    current = diag + tl.where(c1 == c2, match, mismatch)
    current = max(current, up + gap)
    current = max(current, left + gap)
    current = max(current, 0)
    
    tl.store(dp_ptr + i*dp_stride + j, current, mask=mask)

def smith_waterman_gpu(seq1, seq2, match=2, mismatch=-1, gap=-1):
    # 转换为ASCII张量
    seq1_ascii = [ord(c) for c in seq1]
    seq2_ascii = [ord(c) for c in seq2]
    seq1_tensor = torch.tensor(seq1_ascii, device='cuda', dtype=torch.int32)
    seq2_tensor = torch.tensor(seq2_ascii, device='cuda', dtype=torch.int32)
    
    m, n = len(seq1), len(seq2)
    dp = torch.zeros((m+1, n+1), dtype=torch.int32, device='cuda')
    
    max_k = m + n
    for k in range(2, max_k + 1):
        i_min = max(1, k - n)
        i_max = min(m, k - 1)
        if i_min > i_max:
            continue
            
        num_i = i_max - i_min + 1
        grid = lambda meta: (triton.cdiv(num_i, meta['BLOCK_SIZE']),)
        sw_kernel[grid](
            seq1_tensor, seq2_tensor, dp, k,
            m, n, match, mismatch, gap,
            seq1_tensor.stride(0), seq2_tensor.stride(0), dp.stride(0),
            BLOCK_SIZE=256
        )
    
    max_score = torch.max(dp).item()
    max_pos = torch.argmax(dp.view(-1)).item()
    max_i, max_j = max_pos // (n+1), max_pos % (n+1)
    
    return max_score, (max_i, max_j), dp

# ------------------------- 测试案例生成 -------------------------
def generate_test_case(base_len=5000, var_len=1000):
    """生成包含多种特征的复杂测试案例"""
    # 生成基础匹配段
    base = [random.choice('ATCG') for _ in range(base_len)]
    
    # 插入变异
    test1 = base + [random.choice('ATCG') for _ in range(var_len)]
    test2 = base.copy()
    
    # 添加不匹配段
    test2[-100:] = [random.choice('ATCG') for _ in range(100)]
    
    # 添加连续缺口
    test2.insert(300, '-'*50)
    
    # 添加随机噪声
    for _ in range(100):
        idx = random.randint(0, len(test2)-1)
        test2[idx] = random.choice('ATCG')
    
    return (
        ''.join(test1).replace('-', ''),
        ''.join(test2).replace('-', '')
    )

# ------------------------- 测试执行 -------------------------
if __name__ == "__main__":
    # 复杂测试案例
    seq_a, seq_b = generate_test_case()
    print(f"Test case size: {len(seq_a)} vs {len(seq_b)}")
    
    # CPU 测试
    start = time.time()
    cpu_score, cpu_pos, cpu_dp = smith_waterman_cpu(seq_a, seq_b)
    cpu_time = time.time() - start
    
    # GPU 测试
    torch.cuda.synchronize()
    start = time.time()
    gpu_score, gpu_pos, gpu_dp = smith_waterman_gpu(seq_a, seq_b)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    # 结果验证
    assert cpu_score == gpu_score, f"Score mismatch: CPU={cpu_score}, GPU={gpu_score}"
    assert cpu_pos == gpu_pos, f"Position mismatch: CPU={cpu_pos}, GPU={gpu_pos}"
    
    print(f"CPU Time: {cpu_time:.2f}s | GPU Time: {gpu_time:.4f}s")
    print(f"Max Score: {cpu_score} | Position: {cpu_pos}")

        

