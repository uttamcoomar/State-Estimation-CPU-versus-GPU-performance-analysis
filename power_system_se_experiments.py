"""
Power System State Estimation: GPU vs CPU Performance Analysis
Includes decomposed scenarios, GPU utilization, and communication overhead estimation
Designed for Google Colab with Nvidia T4 GPU

MODIFIED VERSION: Works without cupyx.scipy.sparse.linalg.cg
- Implements custom GPU CG solver
- Uses available GPU sparse operations
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, cg
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    GPU_AVAILABLE = True
    print("✓ CuPy loaded successfully!")
except ImportError as e:
    print(f"✗ CuPy import failed: {e}")
    GPU_AVAILABLE = False

# For GPU utilization monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False
    print("pynvml not available. GPU utilization tracking will be limited.")


def cg_gpu_custom(A, b, x0=None, tol=1e-6, maxiter=1000):
    """
    Custom Conjugate Gradient solver for GPU using CuPy
    Implements standard CG algorithm for sparse symmetric positive definite matrices
    
    Args:
        A: Sparse matrix (CuPy CSR format)
        b: Right-hand side vector (CuPy array)
        x0: Initial guess (optional)
        tol: Convergence tolerance
        maxiter: Maximum iterations
    
    Returns:
        x: Solution vector
        info: 0 if converged, 1 if not
    """
    n = len(b)
    x = cp.zeros(n, dtype=b.dtype) if x0 is None else x0.copy()
    r = b - A @ x
    p = r.copy()
    rsold = cp.dot(r, r)
    
    for i in range(maxiter):
        Ap = A @ p
        alpha = rsold / cp.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = cp.dot(r, r)
        
        if cp.sqrt(rsnew) < tol:
            return x, 0  # Converged
        
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x, 1  # Did not converge


class PowerSystemSESimulator:
    """Simulates state estimation for power systems of various sizes"""
    
    def __init__(self, n_buses: int, sparsity: float = 0.95):
        """
        Initialize power system
        
        Args:
            n_buses: Number of buses in the system
            sparsity: Sparsity factor for admittance matrix (0.95 = 95% zeros)
        """
        self.n_buses = n_buses
        self.sparsity = sparsity
        self.n_states = 2 * n_buses  # voltage magnitude and angle for each bus
        
    def generate_measurement_jacobian(self) -> sp.csr_matrix:
        """Generate sparse Jacobian matrix H for SE problem"""
        # Typical SE has 2-3x measurements as states
        n_measurements = int(2.5 * self.n_states)
        
        # Create sparse random Jacobian (mimics real power system topology)
        nnz = int(n_measurements * self.n_states * (1 - self.sparsity))
        
        rows = np.random.randint(0, n_measurements, nnz)
        cols = np.random.randint(0, self.n_states, nnz)
        data = np.random.randn(nnz) * 0.1 + 1.0  # Realistic sensitivity values
        
        H = sp.csr_matrix((data, (rows, cols)), 
                          shape=(n_measurements, self.n_states))
        return H
    
    def generate_se_problem(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Generate weighted least squares SE problem: min ||W^0.5(z - h(x))||^2
        Returns: (H^T W H) matrix and (H^T W z) vector
        """
        H = self.generate_measurement_jacobian()
        n_measurements = H.shape[0]
        
        # Measurement weights (diagonal matrix W)
        weights = np.random.uniform(0.5, 2.0, n_measurements)
        
        # Create gain matrix G = H^T W H
        W_sqrt = sp.diags(np.sqrt(weights))
        H_weighted = W_sqrt @ H
        G = H_weighted.T @ H_weighted
        
        # Create measurement vector
        z = np.random.randn(n_measurements)
        b = H_weighted.T @ (W_sqrt @ z)
        
        return G, b


class PerformanceTracker:
    """Track timing and GPU utilization metrics"""
    
    def __init__(self):
        self.results = []
        if NVML_AVAILABLE:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    def get_gpu_utilization(self) -> Dict[str, float]:
        """Get current GPU utilization"""
        if not NVML_AVAILABLE:
            return {'gpu_util': 0.0, 'memory_util': 0.0}
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                'gpu_util': util.gpu,
                'memory_util': (memory.used / memory.total) * 100
            }
        except:
            return {'gpu_util': 0.0, 'memory_util': 0.0}
    
    def record_result(self, **kwargs):
        """Record experimental result"""
        self.results.append(kwargs)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        return pd.DataFrame(self.results)


def solve_sparse_direct_cpu(G: sp.csr_matrix, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solve using sparse direct method on CPU"""
    start = time.time()
    x = spsolve(G, b)
    elapsed = time.time() - start
    return x, elapsed


def solve_conjugate_gradient_cpu(G: sp.csr_matrix, b: np.ndarray, 
                                  tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Solve using conjugate gradient on CPU"""
    start = time.time()
    x, info = cg(G, b, atol=tol, maxiter=1000)
    elapsed = time.time() - start
    return x, elapsed


def solve_sparse_direct_gpu(G: sp.csr_matrix, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solve using sparse direct method on GPU"""
    if not GPU_AVAILABLE:
        return np.zeros_like(b), float('inf')
    
    try:
        start = time.time()
        
        # Transfer to GPU
        G_gpu = cp_sparse.csr_matrix(G)
        b_gpu = cp.array(b)
        
        # Solve using spsolve if available, otherwise use custom CG
        try:
            from cupyx.scipy.sparse.linalg import spsolve as cp_spsolve
            x_gpu = cp_spsolve(G_gpu, b_gpu)
        except ImportError:
            # Fallback to custom CG solver
            x_gpu, info = cg_gpu_custom(G_gpu, b_gpu)
        
        # Transfer back
        x = cp.asnumpy(x_gpu)
        cp.cuda.Stream.null.synchronize()  # Ensure completion
        
        elapsed = time.time() - start
        return x, elapsed
    except Exception as e:
        print(f"GPU solve error: {e}")
        return np.zeros_like(b), float('inf')


def solve_conjugate_gradient_gpu(G: sp.csr_matrix, b: np.ndarray, 
                                  tol: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Solve using conjugate gradient on GPU with custom implementation"""
    if not GPU_AVAILABLE:
        return np.zeros_like(b), float('inf')
    
    try:
        start = time.time()
        
        # Transfer to GPU
        G_gpu = cp_sparse.csr_matrix(G)
        b_gpu = cp.array(b)
        
        # Use custom CG solver
        x_gpu, info = cg_gpu_custom(G_gpu, b_gpu, tol=tol, maxiter=1000)
        
        # Transfer back
        x = cp.asnumpy(x_gpu)
        cp.cuda.Stream.null.synchronize()
        
        elapsed = time.time() - start
        return x, elapsed
    except Exception as e:
        print(f"GPU CG error: {e}")
        return np.zeros_like(b), float('inf')


# ============================================================================
# EXPERIMENT 1: Decomposed Scenario Analysis
# ============================================================================

def experiment_decomposed_scenarios(tracker: PerformanceTracker):
    """
    Compare: solving N small problems vs 1 large problem
    Tests if decomposition negates GPU benefits
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Decomposed vs Monolithic Problem Solving")
    print("="*70)
    
    scenarios = [
        (4, 1000),   # 4 subproblems of 1000 buses each = 4K total
        (2, 2000),  # 2 subproblems of 2000 buses each = 4K total
        (1, 4000),  # 1 monolithic problem of 4K buses
        (8, 500),  # 8 subproblems of 500 buses each = 4K total
        (10, 400), # 10 subproblems of 400 buses each = 4K total
    ]
    
    for n_subproblems, buses_per_subproblem in scenarios:
        total_buses = n_subproblems * buses_per_subproblem
        print(f"\nScenario: {n_subproblems} × {buses_per_subproblem} buses "
              f"(Total: {total_buses} buses)")
        
        # CPU - Sparse Direct
        cpu_sd_times = []
        for i in range(n_subproblems):
            sim = PowerSystemSESimulator(buses_per_subproblem)
            G, b = sim.generate_se_problem()
            _, t = solve_sparse_direct_cpu(G, b)
            cpu_sd_times.append(t)
        cpu_sd_total = sum(cpu_sd_times)
        
        # CPU - Conjugate Gradient
        cpu_cg_times = []
        for i in range(n_subproblems):
            sim = PowerSystemSESimulator(buses_per_subproblem)
            G, b = sim.generate_se_problem()
            _, t = solve_conjugate_gradient_cpu(G, b)
            cpu_cg_times.append(t)
        cpu_cg_total = sum(cpu_cg_times)
        
        # GPU - Sparse Direct
        gpu_sd_times = []
        gpu_utils = []
        for i in range(n_subproblems):
            sim = PowerSystemSESimulator(buses_per_subproblem)
            G, b = sim.generate_se_problem()
            _, t = solve_sparse_direct_gpu(G, b)
            gpu_sd_times.append(t)
            if NVML_AVAILABLE:
                gpu_utils.append(tracker.get_gpu_utilization()['gpu_util'])
        gpu_sd_total = sum(gpu_sd_times)
        
        # GPU - Conjugate Gradient
        gpu_cg_times = []
        for i in range(n_subproblems):
            sim = PowerSystemSESimulator(buses_per_subproblem)
            G, b = sim.generate_se_problem()
            _, t = solve_conjugate_gradient_gpu(G, b)
            gpu_cg_times.append(t)
        gpu_cg_total = sum(gpu_cg_times)
        
        avg_gpu_util = np.mean(gpu_utils) if gpu_utils else 0.0
        
        # Calculate speedups
        speedup_sd = cpu_sd_total / gpu_sd_total if gpu_sd_total > 0 else 0
        speedup_cg = cpu_cg_total / gpu_cg_total if gpu_cg_total > 0 else 0
        
        print(f"  CPU Sparse Direct:     {cpu_sd_total:.4f}s")
        print(f"  GPU Sparse Direct:     {gpu_sd_total:.4f}s (Speedup: {speedup_sd:.2f}x)")
        print(f"  CPU Conjugate Grad:    {cpu_cg_total:.4f}s")
        print(f"  GPU Conjugate Grad:    {gpu_cg_total:.4f}s (Speedup: {speedup_cg:.2f}x)")
        print(f"  Avg GPU Utilization:   {avg_gpu_util:.1f}%")
        
        # Record results
        tracker.record_result(
            experiment='decomposed',
            n_subproblems=n_subproblems,
            buses_per_subproblem=buses_per_subproblem,
            total_buses=total_buses,
            cpu_sparse_direct=cpu_sd_total,
            cpu_conjugate_gradient=cpu_cg_total,
            gpu_sparse_direct=gpu_sd_total,
            gpu_conjugate_gradient=gpu_cg_total,
            speedup_sd=speedup_sd,
            speedup_cg=speedup_cg,
            avg_gpu_utilization=avg_gpu_util
        )


# ============================================================================
# EXPERIMENT 2: GPU Utilization Analysis
# ============================================================================

def experiment_gpu_utilization(tracker: PerformanceTracker):
    """
    Monitor GPU utilization across different problem sizes
    Focus on the 20K bus scenario
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: GPU Utilization Analysis")
    print("="*70)
    
    bus_sizes = [500, 1000, 1500, 2500, 4000]
    
    for n_buses in bus_sizes:
        print(f"\nAnalyzing {n_buses} buses...")
        
        sim = PowerSystemSESimulator(n_buses)
        G, b = sim.generate_se_problem()
        
        # Warm up GPU
        if GPU_AVAILABLE:
            _ = solve_sparse_direct_gpu(G, b)
            cp.cuda.Stream.null.synchronize()
        
        # Sparse Direct with utilization monitoring
        gpu_utils_sd = []
        memory_utils = []
        times_sd = []
        
        for trial in range(5):  # Multiple trials for averaging
            if NVML_AVAILABLE:
                util_before = tracker.get_gpu_utilization()
            
            _, t = solve_sparse_direct_gpu(G, b)
            times_sd.append(t)
            
            if NVML_AVAILABLE:
                util_after = tracker.get_gpu_utilization()
                gpu_utils_sd.append(max(util_before['gpu_util'], util_after['gpu_util']))
                memory_utils.append(util_after['memory_util'])
        
        # Conjugate Gradient with utilization monitoring
        gpu_utils_cg = []
        times_cg = []
        
        for trial in range(5):
            if NVML_AVAILABLE:
                util_before = tracker.get_gpu_utilization()
            
            _, t = solve_conjugate_gradient_gpu(G, b)
            times_cg.append(t)
            
            if NVML_AVAILABLE:
                util_after = tracker.get_gpu_utilization()
                gpu_utils_cg.append(max(util_before['gpu_util'], util_after['gpu_util']))
        
        avg_util_sd = np.mean(gpu_utils_sd) if gpu_utils_sd else 0.0
        avg_util_cg = np.mean(gpu_utils_cg) if gpu_utils_cg else 0.0
        avg_memory = np.mean(memory_utils) if memory_utils else 0.0
        avg_time_sd = np.mean(times_sd)
        avg_time_cg = np.mean(times_cg)
        
        print(f"  Sparse Direct - Avg GPU Util: {avg_util_sd:.1f}%, "
              f"Memory: {avg_memory:.1f}%, Time: {avg_time_sd:.4f}s")
        print(f"  Conjugate Grad - Avg GPU Util: {avg_util_cg:.1f}%, "
              f"Time: {avg_time_cg:.4f}s")
        
        # Record results
        tracker.record_result(
            experiment='gpu_utilization',
            n_buses=n_buses,
            avg_gpu_util_sparse_direct=avg_util_sd,
            avg_gpu_util_conjugate_gradient=avg_util_cg,
            avg_memory_util=avg_memory,
            avg_time_sparse_direct=avg_time_sd,
            avg_time_conjugate_gradient=avg_time_cg,
            efficiency_sd=avg_util_sd / (avg_time_sd * 1000) if avg_time_sd > 0 else 0,
            efficiency_cg=avg_util_cg / (avg_time_cg * 1000) if avg_time_cg > 0 else 0
        )


# ============================================================================
# EXPERIMENT 3: Communication Overhead Estimation
# ============================================================================

def experiment_communication_overhead(tracker: PerformanceTracker):
    """
    Estimate communication overhead for distributed SE algorithms
    Compare ADMM, Auxiliary Problem Principle, and hierarchical methods
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Communication Overhead in Distributed SE")
    print("="*70)
    
    # Network characteristics (realistic estimates)
    networks = {
        'LAN_DataCenter': {'latency_ms': 0.1, 'bandwidth_mbps': 10000},  # 10 Gbps
        'LAN_Office': {'latency_ms': 1.0, 'bandwidth_mbps': 1000},      # 1 Gbps
        'WAN_Regional': {'latency_ms': 20, 'bandwidth_mbps': 100},      # 100 Mbps
        'WAN_CrossCountry': {'latency_ms': 50, 'bandwidth_mbps': 50},   # 50 Mbps
    }
    
    # Distributed algorithm characteristics
    algorithms = {
        'ADMM': {'iterations': 50, 'data_per_iter_mb': 0.1},  # Dual variable exchange
        'APP': {'iterations': 30, 'data_per_iter_mb': 0.15},   # Auxiliary variable exchange
        'Hierarchical': {'iterations': 20, 'data_per_iter_mb': 0.2},  # Aggregated data
    }
    
    scenarios = [
        (2, 1000),  # 2 regions, 1K buses each
        (4, 500),   # 4 regions, 500 buses each
        (8, 250),   # 8 regions, 250 buses each
        (4, 1000),  # 4 regions, 1K buses each (larger system)
    ]
    
    for n_regions, buses_per_region in scenarios:
        total_buses = n_regions * buses_per_region
        print(f"\n{n_regions} regions × {buses_per_region} buses "
              f"(Total: {total_buses} buses)")
        
        # Compute local solve time (once per iteration)
        sim = PowerSystemSESimulator(buses_per_region)
        G, b = sim.generate_se_problem()
        
        _, cpu_time_sd = solve_sparse_direct_cpu(G, b)
        _, cpu_time_cg = solve_conjugate_gradient_cpu(G, b)
        _, gpu_time_sd = solve_sparse_direct_gpu(G, b)
        _, gpu_time_cg = solve_conjugate_gradient_gpu(G, b)
        
        for algo_name, algo_params in algorithms.items():
            for net_name, net_params in networks.items():
                # Communication time per iteration
                latency = net_params['latency_ms'] / 1000  # Convert to seconds
                bandwidth = net_params['bandwidth_mbps']
                data_size = algo_params['data_per_iter_mb']
                
                # Time for one message exchange (round-trip)
                comm_time_per_iter = 2 * latency + (data_size / bandwidth)
                
                # Total communication time
                total_comm_time = comm_time_per_iter * algo_params['iterations']
                
                # Total computation time
                comp_time_cpu_sd = cpu_time_sd * algo_params['iterations']
                comp_time_cpu_cg = cpu_time_cg * algo_params['iterations']
                comp_time_gpu_sd = gpu_time_sd * algo_params['iterations']
                comp_time_gpu_cg = gpu_time_cg * algo_params['iterations']
                
                # Total execution time (comm + comp)
                total_time_cpu_sd = total_comm_time + comp_time_cpu_sd
                total_time_cpu_cg = total_comm_time + comp_time_cpu_cg
                total_time_gpu_sd = total_comm_time + comp_time_gpu_sd
                total_time_gpu_cg = total_comm_time + comp_time_gpu_cg
                
                # Communication to computation ratios
                comm_comp_ratio_cpu_sd = total_comm_time / comp_time_cpu_sd if comp_time_cpu_sd > 0 else float('inf')
                comm_comp_ratio_gpu_sd = total_comm_time / comp_time_gpu_sd if comp_time_gpu_sd > 0 else float('inf')
                
                # Record results
                tracker.record_result(
                    experiment='communication_overhead',
                    algorithm=algo_name,
                    network_type=net_name,
                    n_regions=n_regions,
                    buses_per_region=buses_per_region,
                    total_buses=total_buses,
                    iterations=algo_params['iterations'],
                    total_comm_time=total_comm_time,
                    comp_time_cpu_sd=comp_time_cpu_sd,
                    comp_time_gpu_sd=comp_time_gpu_sd,
                    total_time_cpu_sd=total_time_cpu_sd,
                    total_time_gpu_sd=total_time_gpu_sd,
                    comm_to_comp_ratio_cpu_sd=comm_comp_ratio_cpu_sd,
                    comm_to_comp_ratio_gpu_sd=comm_comp_ratio_gpu_sd
                )
        
        print(f"  CPU time per solve (SD): {cpu_time_sd:.4f}s")
        print(f"  GPU time per solve (SD): {gpu_time_sd:.4f}s")


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_decomposed_results(df: pd.DataFrame, output_path: str = './'):
    """Plot decomposed scenario results"""
    decomp_df = df[df['experiment'] == 'decomposed'].copy()
    
    if len(decomp_df) == 0:
        print("No decomposed results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Decomposed vs Monolithic Problem Analysis', fontsize=16, fontweight='bold')
    
    # Sort by number of subproblems for better visualization
    decomp_df = decomp_df.sort_values('n_subproblems')
    
    # Plot 1: Execution times
    ax = axes[0, 0]
    x = np.arange(len(decomp_df))
    width = 0.2
    
    ax.bar(x - 1.5*width, decomp_df['cpu_sparse_direct'], width, label='CPU Direct', alpha=0.8)
    ax.bar(x - 0.5*width, decomp_df['gpu_sparse_direct'], width, label='GPU Direct', alpha=0.8)
    ax.bar(x + 0.5*width, decomp_df['cpu_conjugate_gradient'], width, label='CPU CG', alpha=0.8)
    ax.bar(x + 1.5*width, decomp_df['gpu_conjugate_gradient'], width, label='GPU CG', alpha=0.8)
    
    ax.set_xlabel('Decomposition Scenario')
    ax.set_ylabel('Total Execution Time (s)')
    ax.set_title('Execution Time Comparison')
    ax.set_xticks(x)
    labels = [f"{row['n_subproblems']}×{row['buses_per_subproblem']//1000}K" 
              for _, row in decomp_df.iterrows()]
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Speedups
    ax = axes[0, 1]
    ax.plot(decomp_df['n_subproblems'], decomp_df['speedup_sd'], 
            'o-', linewidth=2, markersize=8, label='Sparse Direct')
    ax.plot(decomp_df['n_subproblems'], decomp_df['speedup_cg'], 
            's-', linewidth=2, markersize=8, label='Conjugate Gradient')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlabel('Number of Subproblems')
    ax.set_ylabel('GPU Speedup')
    ax.set_title('GPU Speedup vs Decomposition')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: GPU Utilization
    ax = axes[1, 0]
    ax.bar(x, decomp_df['avg_gpu_utilization'], alpha=0.7, color='green')
    ax.set_xlabel('Decomposition Scenario')
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title('Average GPU Utilization')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Efficiency (Speedup per % GPU utilization)
    ax = axes[1, 1]
    efficiency_sd = decomp_df['speedup_sd'] / (decomp_df['avg_gpu_utilization'] / 100)
    efficiency_cg = decomp_df['speedup_cg'] / (decomp_df['avg_gpu_utilization'] / 100)
    
    ax.plot(decomp_df['n_subproblems'], efficiency_sd, 
            'o-', linewidth=2, markersize=8, label='Sparse Direct')
    ax.plot(decomp_df['n_subproblems'], efficiency_cg, 
            's-', linewidth=2, markersize=8, label='Conjugate Gradient')
    ax.set_xlabel('Number of Subproblems')
    ax.set_ylabel('Efficiency (Speedup per GPU%)')
    ax.set_title('GPU Efficiency vs Decomposition')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}decomposed_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}decomposed_analysis.png")
    plt.close()


def plot_gpu_utilization_results(df: pd.DataFrame, output_path: str = './'):
    """Plot GPU utilization analysis"""
    util_df = df[df['experiment'] == 'gpu_utilization'].copy()
    
    if len(util_df) == 0:
        print("No GPU utilization results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GPU Utilization Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: GPU Utilization vs Problem Size
    ax = axes[0, 0]
    ax.plot(util_df['n_buses']/1000, util_df['avg_gpu_util_sparse_direct'], 
            'o-', linewidth=2, markersize=8, label='Sparse Direct')
    ax.plot(util_df['n_buses']/1000, util_df['avg_gpu_util_conjugate_gradient'], 
            's-', linewidth=2, markersize=8, label='Conjugate Gradient')
    ax.axvline(x=20, color='r', linestyle='--', alpha=0.5, label='20K bus target')
    ax.set_xlabel('Number of Buses (thousands)')
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title('GPU Utilization vs System Size')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Memory Utilization
    ax = axes[0, 1]
    ax.plot(util_df['n_buses']/1000, util_df['avg_memory_util'], 
            'o-', linewidth=2, markersize=8, color='purple')
    ax.axvline(x=20, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Buses (thousands)')
    ax.set_ylabel('GPU Memory Utilization (%)')
    ax.set_title('GPU Memory Usage')
    ax.grid(alpha=0.3)
    
    # Plot 3: Execution Time
    ax = axes[1, 0]
    ax.plot(util_df['n_buses']/1000, util_df['avg_time_sparse_direct'], 
            'o-', linewidth=2, markersize=8, label='Sparse Direct')
    ax.plot(util_df['n_buses']/1000, util_df['avg_time_conjugate_gradient'], 
            's-', linewidth=2, markersize=8, label='Conjugate Gradient')
    ax.axvline(x=20, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Buses (thousands)')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('GPU Execution Time')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Efficiency (utilization per millisecond)
    ax = axes[1, 1]
    ax.plot(util_df['n_buses']/1000, util_df['efficiency_sd'], 
            'o-', linewidth=2, markersize=8, label='Sparse Direct')
    ax.plot(util_df['n_buses']/1000, util_df['efficiency_cg'], 
            's-', linewidth=2, markersize=8, label='Conjugate Gradient')
    ax.axvline(x=20, color='r', linestyle='--', alpha=0.5, label='20K bus threshold')
    ax.set_xlabel('Number of Buses (thousands)')
    ax.set_ylabel('Efficiency (Util%/ms)')
    ax.set_title('GPU Utilization Efficiency')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}gpu_utilization_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}gpu_utilization_analysis.png")
    plt.close()


def plot_communication_overhead_results(df: pd.DataFrame, output_path: str = './'):
    """Plot communication overhead analysis"""
    comm_df = df[df['experiment'] == 'communication_overhead'].copy()
    
    if len(comm_df) == 0:
        print("No communication overhead results to plot")
        return
    
    # Focus on ADMM algorithm for clarity
    admm_df = comm_df[comm_df['algorithm'] == 'ADMM'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Communication Overhead in Distributed SE (ADMM Algorithm)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Communication vs Computation time
    ax = axes[0, 0]
    network_types = admm_df['network_type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(network_types)))
    
    for i, net in enumerate(network_types):
        net_data = admm_df[(admm_df['network_type'] == net) & (admm_df['n_regions'] == 4)]
        if len(net_data) > 0:
            buses = net_data['total_buses'] / 1000
            ax.plot(buses, net_data['total_comm_time'], 'o-', 
                   linewidth=2, markersize=8, label=f'{net} (Comm)', color=colors[i])
            ax.plot(buses, net_data['comp_time_gpu_sd'], 's--', 
                   linewidth=2, markersize=8, label=f'{net} (GPU Comp)', color=colors[i], alpha=0.6)
    
    ax.set_xlabel('Total System Size (thousands of buses)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Communication vs GPU Computation Time (4 regions)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 2: Communication-to-Computation ratios
    ax = axes[0, 1]
    for i, net in enumerate(network_types):
        net_data = admm_df[(admm_df['network_type'] == net) & (admm_df['total_buses'] == 20000)]
        if len(net_data) > 0:
            regions = net_data['n_regions']
            ax.plot(regions, net_data['comm_to_comp_ratio_gpu_sd'], 'o-', 
                   linewidth=2, markersize=8, label=net, color=colors[i])
    
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlabel('Number of Regions')
    ax.set_ylabel('Comm/Comp Ratio')
    ax.set_title('Communication Overhead Ratio (20K buses, GPU)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Total execution time comparison
    ax = axes[1, 0]
    lan_data = admm_df[(admm_df['network_type'] == 'LAN_Office') & (admm_df['total_buses'] == 20000)]
    
    x = np.arange(len(lan_data))
    width = 0.35
    
    if len(lan_data) > 0:
        ax.bar(x - width/2, lan_data['total_time_cpu_sd'], width, 
               label='LAN CPU', alpha=0.8)
        ax.bar(x + width/2, lan_data['total_time_gpu_sd'], width, 
               label='LAN GPU', alpha=0.8)
    
    ax.set_xlabel('Number of Regions')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Total Time: CPU vs GPU (LAN, 20K buses)')
    ax.set_xticks(x)
    ax.set_xticklabels(lan_data['n_regions'].values if len(lan_data) > 0 else [])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Network impact on GPU advantage
    ax = axes[1, 1]
    for buses in [10000, 20000]:
        bus_data = admm_df[(admm_df['total_buses'] == buses) & (admm_df['n_regions'] == 4)]
        if len(bus_data) > 0:
            nets = bus_data['network_type']
            speedup = bus_data['total_time_cpu_sd'] / bus_data['total_time_gpu_sd']
            ax.plot(range(len(nets)), speedup, 'o-', 
                   linewidth=2, markersize=8, label=f'{buses} buses')
    
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlabel('Network Type')
    ax.set_ylabel('Overall GPU Speedup')
    ax.set_title('GPU Advantage with Communication (4 regions)')
    ax.set_xticks(range(len(network_types)))
    ax.set_xticklabels(network_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_path}communication_overhead_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}communication_overhead_analysis.png")
    plt.close()


def generate_summary_tables(df: pd.DataFrame, output_path: str = './'):
    """Generate summary tables for paper inclusion"""
    
    # Table 1: Decomposed scenarios summary
    decomp_df = df[df['experiment'] == 'decomposed'].copy()
    if len(decomp_df) > 0:
        table1 = decomp_df[['n_subproblems', 'buses_per_subproblem', 'total_buses',
                            'cpu_sparse_direct', 'gpu_sparse_direct', 'speedup_sd',
                            'avg_gpu_utilization']].copy()
        table1.columns = ['# Subproblems', 'Buses/Subproblem', 'Total Buses',
                         'CPU Time (s)', 'GPU Time (s)', 'Speedup', 'GPU Util (%)']
        table1.to_csv(f'{output_path}table1_decomposed_scenarios.csv', index=False, float_format='%.4f')
        print(f"\nTable 1 saved: {output_path}table1_decomposed_scenarios.csv")
        print(table1.to_string(index=False))
    
    # Table 2: GPU utilization summary
    util_df = df[df['experiment'] == 'gpu_utilization'].copy()
    if len(util_df) > 0:
        table2 = util_df[['n_buses', 'avg_gpu_util_sparse_direct', 
                         'avg_memory_util', 'avg_time_sparse_direct']].copy()
        table2.columns = ['# Buses', 'GPU Util (%)', 'Memory Util (%)', 'Time (s)']
        table2.to_csv(f'{output_path}table2_gpu_utilization.csv', index=False, float_format='%.4f')
        print(f"\nTable 2 saved: {output_path}table2_gpu_utilization.csv")
        print(table2.to_string(index=False))
    
    # Table 3: Communication overhead key findings
    comm_df = df[df['experiment'] == 'communication_overhead'].copy()
    if len(comm_df) > 0:
        # Filter for representative scenarios
        table3 = comm_df[(comm_df['algorithm'] == 'ADMM') & 
                        (comm_df['total_buses'] == 20000)].copy()
        table3 = table3[['n_regions', 'network_type', 'total_comm_time', 
                        'comp_time_gpu_sd', 'comm_to_comp_ratio_gpu_sd']].copy()
        table3.columns = ['# Regions', 'Network', 'Comm Time (s)', 
                         'GPU Comp Time (s)', 'Comm/Comp Ratio']
        table3.to_csv(f'{output_path}table3_communication_overhead.csv', index=False, float_format='%.4f')
        print(f"\nTable 3 saved: {output_path}table3_communication_overhead.csv")
        print(table3.to_string(index=False))


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all experiments and generate visualizations"""
    
    print("="*70)
    print("Power System State Estimation: GPU vs CPU Analysis")
    print("Extended Experiments for Distributed Algorithm Discussion")
    print("="*70)
    
    if not GPU_AVAILABLE:
        print("\nWARNING: CuPy not available. GPU experiments will be skipped.")
        print("Install CuPy with: !pip install cupy-cuda12x")
        return
    
    # Initialize tracker
    tracker = PerformanceTracker()
    
    # Run experiments
    experiment_decomposed_scenarios(tracker)
    experiment_gpu_utilization(tracker)
    experiment_communication_overhead(tracker)
    
    # Get results
    results_df = tracker.to_dataframe()
    
    # Save raw results
    results_df.to_csv('./all_experimental_results.csv', index=False)
    print(f"\n\nRaw results saved: ./all_experimental_results.csv")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("Generating Visualizations...")
    print("="*70)
    
    plot_decomposed_results(results_df)
    plot_gpu_utilization_results(results_df)
    plot_communication_overhead_results(results_df)
    
    # Generate summary tables
    print("\n" + "="*70)
    print("Generating Summary Tables...")
    print("="*70)
    generate_summary_tables(results_df)
    
    print("\n" + "="*70)
    print("All experiments completed successfully!")
    print("="*70)
    print(f"\nGenerated files in ./:")
    print("  - all_experimental_results.csv")
    print("  - decomposed_analysis.png")
    print("  - gpu_utilization_analysis.png")
    print("  - communication_overhead_analysis.png")
    print("  - table1_decomposed_scenarios.csv")
    print("  - table2_gpu_utilization.csv")
    print("  - table3_communication_overhead.csv")


if __name__ == "__main__":
    main()
