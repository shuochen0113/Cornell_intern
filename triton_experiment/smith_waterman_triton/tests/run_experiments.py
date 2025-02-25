import os
import json
import time
import random
import torch
from datetime import datetime
from pathlib import Path
from src import cpu_sw, gpu_sw

def generate_test_case(base_len=5000, var_len=1000):
    base = [random.choice('ATCG') for _ in range(base_len)]
    test1 = base + [random.choice('ATCG') for _ in range(var_len)]
    test2 = base.copy()
    
    test2[-100:] = [random.choice('ATCG') for _ in range(100)]
    test2.insert(300, '-'*50)
    
    for _ in range(100):
        idx = random.randint(0, len(test2)-1)
        test2[idx] = random.choice('ATCG')
    
    return (
        ''.join(test1).replace('-', ''),
        ''.join(test2).replace('-', '')
    )

def traceback(dp, seq1, seq2, pos):
    i, j = pos
    alignment1, alignment2 = [], []
    
    while i > 0 and j > 0 and dp[i][j] > 0:
        current = dp[i][j]
        diag = dp[i-1][j-1]
        up = dp[i-1][j]
        left = dp[i][j-1]
        
        if current == diag + (2 if seq1[i-1] == seq2[j-1] else -1):
            alignment1.append(seq1[i-1])
            alignment2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif current == up -1:
            alignment1.append(seq1[i-1])
            alignment2.append('-')
            i -= 1
        else:
            alignment1.append('-')
            alignment2.append(seq2[j-1])
            j -= 1
    
    return ''.join(reversed(alignment1)), ''.join(reversed(alignment2))

def run_experiment(test_case):
    seq_a, seq_b = test_case
    
    # CPU测试
    cpu_start = time.time()
    cpu_score, cpu_pos, cpu_dp = cpu_sw.smith_waterman_cpu(seq_a, seq_b)
    cpu_time = time.time() - cpu_start
    
    # GPU测试
    torch.cuda.synchronize()
    gpu_start = time.time()
    gpu_score, gpu_pos, gpu_dp = gpu_sw.smith_waterman_gpu(seq_a, seq_b)
    torch.cuda.synchronize()
    gpu_time = time.time() - gpu_start
    
    # 回溯
    alignment_a, alignment_b = traceback(cpu_dp, seq_a, seq_b, cpu_pos)
    
    return {
        "lengths": (len(seq_a), len(seq_b)),
        "cpu_time": cpu_time,
        "gpu_time": gpu_time,
        "score": cpu_score,
        "position": cpu_pos,
        "alignment": (alignment_a, alignment_b),
        "speedup": cpu_time / gpu_time if gpu_time != 0 else 0
    }

def main():
    test_cases = [
        (100, 20),    # 小规模
        (500, 100),   # 中小规模
        (2000, 500),  # 中等规模
        (5000, 1000), # 大规模
        (10000, 2000) # 超大规模
    ]
    
    results = []
    for base_len, var_len in test_cases:
        print(f"Running experiment with base_len={base_len}, var_len={var_len}")
        test_case = generate_test_case(base_len, var_len)
        result = run_experiment(test_case)
        results.append(result)
    
    # 保存结果
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()