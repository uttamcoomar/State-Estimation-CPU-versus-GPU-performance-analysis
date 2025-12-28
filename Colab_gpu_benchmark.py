"""
Power System State Estimation GPU Benchmark for Google Colab
Optimized for T4 GPU with realistic test cases
"""
# DSSE algorithms

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, cg
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator

# GPU libraries
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpsplinalg

print("="*80)
print("Power System State Estimation Benchmark - GPU Accelerated")
print("="*80)

# Verify GPU
print(f"\n✓ GPU Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
print(f"✓ GPU Memory: {cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9:.1f} GB")
print(f"✓ CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")

def generate_power_system_matrix(n_buses, avg_connections=4):
    """
    Generate realistic sparse symmetric positive definite matrix
    simulating power system gain matrix (H^T * W * H)
    
    Structure mimics actual power networks:
    - Sparse connectivity (each bus connects to ~3-5 neighbors)
    - Symmetric positive definite
    - Condition number increases with system size
    """
    np.random.seed(42 + n_buses)  # Reproducible but different per size
    
    # Build adjacency structure
    row_idx = []
    col_idx = []
    values = []
    
    # Diagonal elements (self-admittance)
    for i in range(n_buses):
        diag_value = 10.0 + np.random.rand() * 5.0
        row_idx.append(i)
        col_idx.append(i)
        values.append(diag_value)
    
    # Off-diagonal (line connections)
    for i in range(n_buses):
        # Each bus connects to avg_connections neighbors
        n_neighbors = np.random.randint(max(2, avg_connections-2), avg_connections+2)
        
        # Prefer nearby buses (mimics geographical proximity)
        possible_neighbors = list(range(max(0, i-20), min(n_buses, i+20)))
        if i in possible_neighbors:
            possible_neighbors.remove(i)
        
        if len(possible_neighbors) > 0:
            n_select = min(n_neighbors, len(possible_neighbors))
            neighbors = np.random.choice(possible_neighbors, n_select, replace=False)
            
            for j in neighbors:
                # Line admittance (negative off-diagonal)
                value = -(0.5 + np.random.rand() * 2.0)
                
                row_idx.append(i)
                col_idx.append(j)
                values.append(value)
                
                # Symmetric
                row_idx.append(j)
                col_idx.append(i)
                values.append(value)
    
    # Create sparse matrix
    A = sp.coo_matrix((values, (row_idx, col_idx)), shape=(n_buses, n_buses))
    A = A.tocsr()
    
    # Ensure symmetric positive definite
    A = (A + A.T) / 2  # Symmetrize
    
    # Add diagonal dominance for SPD property
    min_diag = abs(A.sum(axis=1)).A1.max() + 1.0
    A = A + sp.eye(n_buses) * min_diag
    
    return A

def benchmark_cpu_sparse_direct(A, b, num_runs=3):
    """CPU Sparse Direct Solver (scipy spsolve - uses UMFPACK/SuperLU)"""
    times = []
    
    # Warm-up
    _ = spsolve(A, b)
    
    for _ in range(num_runs):
        start = time.perf_counter()
        x = spsolve(A, b)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.median(times), x

def benchmark_cpu_cg_precond(A, b, tol=1e-6, num_runs=3):
    """CPU CG with diagonal (Jacobi) preconditioning"""
    times = []
    iterations_list = []
    
    # Diagonal preconditioner
    M_diag = 1.0 / A.diagonal()
    M = LinearOperator((A.shape[0], A.shape[0]), 
                       matvec=lambda x: M_diag * x)
    
    # Warm-up
    _, _ = cg(A, b, M=M, atol=tol, maxiter=1000)
    
    # Track iterations with callback
    class IterationCounter:
        def __init__(self):
            self.count = 0
        def __call__(self, xk):
            self.count += 1
    
    for _ in range(num_runs):
        counter = IterationCounter()
        start = time.perf_counter()
        x, info = cg(A, b, M=M, atol=tol, maxiter=1000, callback=counter)
        end = time.perf_counter()
        
        if info == 0:  # Converged
            times.append(end - start)
            iterations_list.append(counter.count)
    
    if times:
        return np.median(times), np.median(iterations_list), x
    else:
        return None, None, None

def benchmark_gpu_sparse_direct(A_cpu, b_cpu, num_runs=3):
    """GPU Sparse Direct Solver (cuSOLVER via CuPy)"""
    times = []
    
    # Transfer to GPU
    A_gpu = cpsp.csr_matrix(A_cpu)
    b_gpu = cp.array(b_cpu)
    
    # Warm-up
    _ = cpsplinalg.spsolve(A_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()
    
    for _ in range(num_runs):
        start = time.perf_counter()
        x_gpu = cpsplinalg.spsolve(A_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    x_cpu = cp.asnumpy(x_gpu)
    return np.median(times), x_cpu

def benchmark_gpu_cg(A_cpu, b_cpu, tol=1e-6, num_runs=3):
    """GPU Conjugate Gradient"""
    times = []
    iterations_list = []
    
    A_gpu = cpsp.csr_matrix(A_cpu)
    b_gpu = cp.array(b_cpu)
    
    # Warm-up
    _, _ = cpsplinalg.cg(A_gpu, b_gpu, atol=tol, maxiter=1000)
    cp.cuda.Stream.null.synchronize()
    
    for _ in range(num_runs):
        start = time.perf_counter()
        x_gpu, info = cpsplinalg.cg(A_gpu, b_gpu, atol=tol, maxiter=1000)
        cp.cuda.Stream.null.synchronize()
        end = time.perf_counter()
        
        if info == 0:
            times.append(end - start)
            # Note: CuPy CG doesn't directly return iteration count
    
    if times:
        x_cpu = cp.asnumpy(x_gpu)
        return np.median(times), x_cpu
    else:
        return None, None

def run_comprehensive_benchmark():
    """Run complete benchmark suite"""
    
    # Test system sizes (buses)
    system_sizes = [100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 40000]
    
    results = []
    
    print("\n" + "="*80)
    print("RUNNING BENCHMARKS")
    print("="*80)
    
    for n in system_sizes:
        print(f"\n{'='*80}")
        print(f"System Size: {n:,} buses")
        print(f"{'='*80}")
        
        # Generate problem
        print("Generating sparse matrix...", end=" ")
        A = generate_power_system_matrix(n)
        b = np.random.randn(n)
        
        nnz = A.nnz
        sparsity = 1.0 - (nnz / (n * n))
        density = nnz / n
        
        print(f"✓")
        print(f"  NNZ: {nnz:,} ({sparsity*100:.2f}% sparse)")
        print(f"  Avg connections/bus: {density:.1f}")
        
        result = {
            'n_buses': n,
            'nnz': nnz,
            'sparsity': sparsity,
            'avg_degree': density
        }
        
        # CPU Sparse Direct
        if n <= 10000:  # Memory limit
            try:
                print("\n  CPU Sparse Direct...", end=" ")
                t, x_ref = benchmark_cpu_sparse_direct(A, b)
                result['cpu_direct_time'] = t
                result['cpu_direct_success'] = True
                print(f"✓ {t:.4f}s")
            except Exception as e:
                print(f"✗ Failed: {e}")
                result['cpu_direct_success'] = False
                x_ref = spsolve(A, b)  # Get reference solution anyway
        else:
            result['cpu_direct_success'] = False
            print("\n  CPU Sparse Direct: Skipped (too large)")
            # Need reference solution
            x_ref, _ = cg(A, b, atol=1e-6, maxiter=1000)
        
        # CPU CG with preconditioning
        try:
            print("  CPU CG (preconditioned)...", end=" ")
            t, iters, x = benchmark_cpu_cg_precond(A, b)
            if t is not None:
                result['cpu_cg_time'] = t
                result['cpu_cg_iterations'] = int(iters)
                result['cpu_cg_success'] = True
                
                # Verify accuracy
                residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
                result['cpu_cg_residual'] = residual
                print(f"✓ {t:.4f}s ({int(iters)} iters, residual: {residual:.2e})")
            else:
                print(f"✗ Did not converge")
                result['cpu_cg_success'] = False
        except Exception as e:
            print(f"✗ Failed: {e}")
            result['cpu_cg_success'] = False
        
        # GPU Sparse Direct
        if n <= 15000:  # GPU memory limit
            try:
                print("  GPU Sparse Direct...", end=" ")
                t, x = benchmark_gpu_sparse_direct(A, b)
                result['gpu_direct_time'] = t
                result['gpu_direct_success'] = True
                
                # Verify accuracy
                residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
                result['gpu_direct_residual'] = residual
                
                # Speedup vs CPU
                if 'cpu_direct_time' in result:
                    speedup = result['cpu_direct_time'] / t
                    result['speedup_gpu_vs_cpu_direct'] = speedup
                    print(f"✓ {t:.4f}s (speedup: {speedup:.2f}x, residual: {residual:.2e})")
                else:
                    print(f"✓ {t:.4f}s (residual: {residual:.2e})")
            except Exception as e:
                print(f"✗ Failed: {e}")
                result['gpu_direct_success'] = False
        else:
            result['gpu_direct_success'] = False
            print("  GPU Sparse Direct: Skipped (too large)")
        
        # GPU CG
        try:
            print("  GPU CG...", end=" ")
            t, x = benchmark_gpu_cg(A, b)
            if t is not None:
                result['gpu_cg_time'] = t
                result['gpu_cg_success'] = True
                
                # Verify accuracy
                residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
                result['gpu_cg_residual'] = residual
                
                # Speedup vs CPU CG
                if 'cpu_cg_time' in result:
                    speedup = result['cpu_cg_time'] / t
                    result['speedup_gpu_vs_cpu_cg'] = speedup
                    print(f"✓ {t:.4f}s (speedup: {speedup:.2f}x, residual: {residual:.2e})")
                else:
                    print(f"✓ {t:.4f}s (residual: {residual:.2e})")
            else:
                print(f"✗ Did not converge")
                result['gpu_cg_success'] = False
        except Exception as e:
            print(f"✗ Failed: {e}")
            result['gpu_cg_success'] = False
        
        results.append(result)
    
    return pd.DataFrame(results)

def generate_publication_plots(df):
    """Generate high-quality plots for paper"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot 1: Computation Time Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    methods = [
        ('cpu_direct_time', 'CPU Sparse Direct', 'o-', '#e74c3c', 2.5),
        ('cpu_cg_time', 'CPU CG (Jacobi precond.)', '^-', '#2ecc71', 2.5),
        ('gpu_direct_time', 'GPU Sparse Direct', 's-', '#3498db', 2.5),
        ('gpu_cg_time', 'GPU CG', 'd-', '#9b59b6', 2.5),
    ]
    
    for col, label, style, color, lw in methods:
        if col in df.columns:
            valid = df[col].notna()
            if valid.sum() > 0:
                ax.plot(df[valid]['n_buses'], df[valid][col], 
                       style, label=label, linewidth=lw, markersize=10, color=color)
    
    ax.set_xlabel('Number of Buses', fontsize=14, fontweight='bold')
    ax.set_ylabel('Solution Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('GPU vs CPU Performance: Power System State Estimation\n(Google Colab T4 GPU)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig('benchmark_time_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: benchmark_time_comparison.png")
    plt.show()
    
    # Plot 2: GPU Speedup Analysis
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if 'speedup_gpu_vs_cpu_direct' in df.columns:
        valid = df['speedup_gpu_vs_cpu_direct'].notna()
        if valid.sum() > 0:
            ax.plot(df[valid]['n_buses'], df[valid]['speedup_gpu_vs_cpu_direct'],
                   'o-', label='GPU vs CPU (Direct Solver)', 
                   linewidth=2.5, markersize=10, color='#3498db')
    
    if 'speedup_gpu_vs_cpu_cg' in df.columns:
        valid = df['speedup_gpu_vs_cpu_cg'].notna()
        if valid.sum() > 0:
            ax.plot(df[valid]['n_buses'], df[valid]['speedup_gpu_vs_cpu_cg'],
                   '^-', label='GPU vs CPU (Iterative CG)', 
                   linewidth=2.5, markersize=10, color='#2ecc71')
    
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.6, label='No speedup')
    ax.set_xlabel('Number of Buses', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup Factor (×)', fontsize=14, fontweight='bold')
    ax.set_title('GPU Acceleration Speedup Over CPU\n(Google Colab T4 GPU)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig('benchmark_speedup.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: benchmark_speedup.png")
    plt.show()
    
    # Plot 3: CG Convergence
    if 'cpu_cg_iterations' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        valid = df['cpu_cg_iterations'].notna()
        if valid.sum() > 0:
            ax.plot(df[valid]['n_buses'], df[valid]['cpu_cg_iterations'],
                   'o-', label='Iterations to Convergence', 
                   linewidth=2.5, markersize=10, color='#e74c3c')
        
        ax.set_xlabel('Number of Buses', fontsize=14, fontweight='bold')
        ax.set_ylabel('CG Iterations', fontsize=14, fontweight='bold')
        ax.set_title('Conjugate Gradient Convergence (Jacobi Preconditioned)\nTolerance: 1e-6', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.legend(fontsize=12, framealpha=0.9)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.savefig('benchmark_iterations.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: benchmark_iterations.png")
        plt.show()

def generate_latex_table(df):
    """Generate LaTeX table for paper"""
    
    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{GPU vs CPU Performance for State Estimation Inner Loop Solution (Google Colab T4 GPU)}")
    latex.append("\\label{tab:gpu_benchmark}")
    latex.append("\\begin{tabular}{|r|r|r|r|r|r|r|}")
    latex.append("\\hline")
    latex.append("\\textbf{Buses} & \\textbf{NNZ} & \\textbf{CPU Direct} & \\textbf{GPU Direct} & \\textbf{CPU CG} & \\textbf{GPU CG} & \\textbf{Speedup} \\\\")
    latex.append("& & \\textbf{(s)} & \\textbf{(s)} & \\textbf{(s)} & \\textbf{(s)} & \\textbf{(GPU/CPU)} \\\\")
    latex.append("\\hline")
    
    for _, row in df.iterrows():
        n = int(row['n_buses'])
        nnz = int(row['nnz'])
        
        cpu_d = f"{row['cpu_direct_time']:.4f}" if pd.notna(row.get('cpu_direct_time')) else "---"
        gpu_d = f"{row['gpu_direct_time']:.4f}" if pd.notna(row.get('gpu_direct_time')) else "---"
        cpu_cg = f"{row['cpu_cg_time']:.4f}" if pd.notna(row.get('cpu_cg_time')) else "---"
        gpu_cg = f"{row['gpu_cg_time']:.4f}" if pd.notna(row.get('gpu_cg_time')) else "---"
        
        speedup = ""
        if pd.notna(row.get('speedup_gpu_vs_cpu_direct')):
            speedup = f"{row['speedup_gpu_vs_cpu_direct']:.2f}$\\times$"
        elif pd.notna(row.get('speedup_gpu_vs_cpu_cg')):
            speedup = f"{row['speedup_gpu_vs_cpu_cg']:.2f}$\\times$"
        
        latex.append(f"{n:,} & {nnz:,} & {cpu_d} & {gpu_d} & {cpu_cg} & {gpu_cg} & {speedup} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")
    
    return "\n".join(latex)

def main():
    """Main benchmark execution"""
    
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE GPU BENCHMARK")
    print("="*80)
    print(f"Date: {pd.Timestamp.now()}")
    print(f"Platform: Google Colab")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    
    # Run benchmarks
    df = run_comprehensive_benchmark()
    
    # Save results
    df.to_csv('gpu_benchmark_results.csv', index=False)
    print("\n✓ Results saved: gpu_benchmark_results.csv")
    
    # Generate plots
    print("\nGenerating plots...")
    generate_publication_plots(df)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(df)
    with open('gpu_benchmark_table.tex', 'w') as f:
        f.write(latex_table)
    print("\n✓ LaTeX table saved: gpu_benchmark_table.tex")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    print("\nDetailed Results:")
    print(df.to_string())
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    if 'speedup_gpu_vs_cpu_direct' in df.columns:
        valid = df['speedup_gpu_vs_cpu_direct'].notna()
        if valid.sum() > 0:
            max_speedup = df[valid]['speedup_gpu_vs_cpu_direct'].max()
            max_speedup_n = df[valid].loc[df[valid]['speedup_gpu_vs_cpu_direct'].idxmax(), 'n_buses']
            print(f"\n✓ Maximum speedup (Direct solver): {max_speedup:.2f}× at {int(max_speedup_n):,} buses")
    
    if 'speedup_gpu_vs_cpu_cg' in df.columns:
        valid = df['speedup_gpu_vs_cpu_cg'].notna()
        if valid.sum() > 0:
            max_speedup = df[valid]['speedup_gpu_vs_cpu_cg'].max()
            max_speedup_n = df[valid].loc[df[valid]['speedup_gpu_vs_cpu_cg'].idxmax(), 'n_buses']
            print(f"✓ Maximum speedup (CG solver): {max_speedup:.2f}× at {int(max_speedup_n):,} buses")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print("\nFiles generated:")
    print("  • gpu_benchmark_results.csv")
    print("  • benchmark_time_comparison.png")
    print("  • benchmark_speedup.png")
    print("  • benchmark_iterations.png")
    print("  • gpu_benchmark_table.tex")
    print("\nYou can now download these files and use them in your paper!")

if __name__ == "__main__":
    main()