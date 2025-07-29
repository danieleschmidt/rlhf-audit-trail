#!/usr/bin/env python3
"""Performance benchmark runner for RLHF Audit Trail.

This script runs comprehensive performance benchmarks to measure:
- Audit logging overhead
- Privacy computation performance
- Model checkpoint performance
- Database operation latency
- Memory usage patterns
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import statistics
import psutil
import os

# Benchmark configuration
BENCHMARK_CONFIG = {
    "iterations": 100,
    "warmup_iterations": 10,
    "timeout_seconds": 300,
    "memory_monitoring": True,
    "cpu_monitoring": True,
}

def measure_memory_usage() -> Dict[str, float]:
    """Measure current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent(),
    }

def measure_cpu_usage() -> float:
    """Measure current CPU usage."""
    return psutil.cpu_percent(interval=0.1)

class BenchmarkRunner:
    """Runs and collects performance benchmarks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: Dict[str, Any] = {
            "timestamp": time.time(),
            "system_info": self._get_system_info(),
            "benchmarks": {},
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "python_version": os.sys.version,
            "platform": os.sys.platform,
        }
    
    def benchmark_audit_logging(self) -> Dict[str, Any]:
        """Benchmark audit logging performance."""
        print("🔍 Benchmarking audit logging performance...")
        
        # Mock audit logging function
        def mock_audit_log(data: Dict[str, Any]) -> None:
            """Mock audit log function."""
            # Simulate serialization overhead
            json.dumps(data)
            # Simulate hash computation
            hash(str(data))
        
        # Benchmark data
        test_data = {
            "event_type": "annotation",
            "timestamp": time.time(),
            "data": {"prompt": "x" * 1000, "response": "y" * 1000},
            "metadata": {"user_id": "test_user", "session_id": "test_session"},
        }
        
        times = []
        memory_usage = []
        
        # Warmup
        for _ in range(self.config["warmup_iterations"]):
            mock_audit_log(test_data)
        
        # Actual benchmark
        for i in range(self.config["iterations"]):
            start_memory = measure_memory_usage() if self.config["memory_monitoring"] else None
            
            start_time = time.perf_counter()
            mock_audit_log(test_data)
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            if self.config["memory_monitoring"] and i % 10 == 0:
                memory_usage.append(measure_memory_usage())
        
        return {
            "operation": "audit_logging",
            "iterations": self.config["iterations"],
            "avg_time_ms": statistics.mean(times),
            "median_time_ms": statistics.median(times),
            "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "p95_time_ms": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
            "memory_usage": memory_usage if memory_usage else None,
        }
    
    def benchmark_privacy_computation(self) -> Dict[str, Any]:
        """Benchmark differential privacy computation."""
        print("🔒 Benchmarking privacy computation performance...")
        
        # Mock privacy computation
        def mock_dp_noise(epsilon: float, sensitivity: float) -> float:
            """Mock differential privacy noise generation."""
            import random
            return random.gauss(0, sensitivity / epsilon)
        
        times = []
        
        # Warmup
        for _ in range(self.config["warmup_iterations"]):
            mock_dp_noise(1.0, 1.0)
        
        # Benchmark
        for _ in range(self.config["iterations"]):
            start_time = time.perf_counter()
            for _ in range(100):  # Batch operations
                mock_dp_noise(1.0, 1.0)
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)
        
        return {
            "operation": "privacy_computation",
            "batch_size": 100,
            "iterations": self.config["iterations"],
            "avg_time_ms": statistics.mean(times),
            "median_time_ms": statistics.median(times),
            "throughput_ops_per_sec": (100 * self.config["iterations"]) / (sum(times) / 1000),
        }
    
    def benchmark_model_checkpoint(self) -> Dict[str, Any]:
        """Benchmark model checkpoint operations."""
        print("💾 Benchmarking model checkpoint performance...")
        
        # Mock checkpoint data (simulate model weights)
        checkpoint_data = {
            "model_state": {"layer_" + str(i): [0.1] * 1000 for i in range(10)},
            "optimizer_state": {"param_" + str(i): [0.01] * 100 for i in range(10)},
            "metadata": {
                "epoch": 1,
                "step": 1000,
                "loss": 0.5,
                "timestamp": time.time(),
            }
        }
        
        times = []
        sizes = []
        
        # Warmup
        for _ in range(self.config["warmup_iterations"]):
            serialized = json.dumps(checkpoint_data)
        
        # Benchmark
        for _ in range(min(self.config["iterations"], 50)):  # Reduce iterations for heavy operations
            start_time = time.perf_counter()
            serialized = json.dumps(checkpoint_data)
            # Simulate compression
            compressed_size = len(serialized.encode('utf-8'))
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)
            sizes.append(compressed_size)
        
        return {
            "operation": "model_checkpoint",
            "iterations": min(self.config["iterations"], 50),
            "avg_time_ms": statistics.mean(times),
            "avg_size_bytes": statistics.mean(sizes),
            "throughput_mb_per_sec": (statistics.mean(sizes) / 1024 / 1024) / (statistics.mean(times) / 1000),
        }
    
    def benchmark_database_operations(self) -> Dict[str, Any]:
        """Benchmark database operation simulation."""
        print("🗄️ Benchmarking database operations...")
        
        # Mock database operations
        mock_db = {}
        operation_times = {"insert": [], "select": [], "update": []}
        
        # Generate test data
        test_records = [
            {"id": i, "data": f"record_{i}", "timestamp": time.time()}
            for i in range(1000)
        ]
        
        # Insert benchmark
        for record in test_records[:self.config["iterations"]]:
            start_time = time.perf_counter()
            mock_db[record["id"]] = record
            end_time = time.perf_counter()
            operation_times["insert"].append((end_time - start_time) * 1000)
        
        # Select benchmark
        for i in range(min(self.config["iterations"], len(mock_db))):
            start_time = time.perf_counter()
            _ = mock_db.get(i)
            end_time = time.perf_counter()
            operation_times["select"].append((end_time - start_time) * 1000)
        
        # Update benchmark
        for i in range(min(self.config["iterations"], len(mock_db))):
            start_time = time.perf_counter()
            if i in mock_db:
                mock_db[i]["updated"] = True
            end_time = time.perf_counter()
            operation_times["update"].append((end_time - start_time) * 1000)
        
        return {
            "operation": "database_operations",
            "record_count": len(mock_db),
            "insert_avg_ms": statistics.mean(operation_times["insert"]),
            "select_avg_ms": statistics.mean(operation_times["select"]),
            "update_avg_ms": statistics.mean(operation_times["update"]),
            "total_operations": sum(len(times) for times in operation_times.values()),
        }
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and collect results."""
        print("🚀 Starting comprehensive performance benchmarks...")
        print(f"Configuration: {self.config}")
        print("-" * 60)
        
        benchmarks = [
            self.benchmark_audit_logging,
            self.benchmark_privacy_computation,
            self.benchmark_model_checkpoint,
            self.benchmark_database_operations,
        ]
        
        for benchmark in benchmarks:
            try:
                result = benchmark()
                self.results["benchmarks"][result["operation"]] = result
            except Exception as e:
                print(f"❌ Benchmark {benchmark.__name__} failed: {e}")
                self.results["benchmarks"][benchmark.__name__] = {"error": str(e)}
        
        # System resource summary
        self.results["system_resources"] = {
            "final_memory": measure_memory_usage(),
            "final_cpu": measure_cpu_usage(),
        }
        
        return self.results
    
    def save_results(self, output_path: Path) -> None:
        """Save benchmark results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📊 Results saved to: {output_path}")
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("📈 BENCHMARK SUMMARY")
        print("=" * 60)
        
        for name, result in self.results["benchmarks"].items():
            if "error" in result:
                print(f"❌ {name}: ERROR - {result['error']}")
                continue
            
            print(f"\n🔧 {name.upper().replace('_', ' ')}")
            print("-" * 40)
            
            if "avg_time_ms" in result:
                print(f"  Average Time: {result['avg_time_ms']:.3f} ms")
            if "median_time_ms" in result:
                print(f"  Median Time:  {result['median_time_ms']:.3f} ms")
            if "throughput_ops_per_sec" in result:
                print(f"  Throughput:   {result['throughput_ops_per_sec']:.0f} ops/sec")
            if "throughput_mb_per_sec" in result:
                print(f"  Throughput:   {result['throughput_mb_per_sec']:.2f} MB/sec")
            
            # Operation-specific metrics
            if name == "database_operations":
                print(f"  Insert Avg:   {result['insert_avg_ms']:.3f} ms")
                print(f"  Select Avg:   {result['select_avg_ms']:.3f} ms")
                print(f"  Update Avg:   {result['update_avg_ms']:.3f} ms")
        
        # System resources
        memory = self.results["system_resources"]["final_memory"]
        print(f"\n💻 SYSTEM RESOURCES")
        print("-" * 40)
        print(f"  Memory Usage: {memory['rss_mb']:.1f} MB ({memory['percent']:.1f}%)")
        print(f"  CPU Usage:    {self.results['system_resources']['final_cpu']:.1f}%")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run RLHF Audit Trail performance benchmarks")
    parser.add_argument("--iterations", type=int, default=BENCHMARK_CONFIG["iterations"],
                       help="Number of benchmark iterations")
    parser.add_argument("--output", type=Path, default="benchmarks/results/benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--config", type=Path, help="Custom benchmark configuration file")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks (fewer iterations)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = BENCHMARK_CONFIG.copy()
    if args.config and args.config.exists():
        with open(args.config) as f:
            config.update(json.load(f))
    
    if args.quick:
        config["iterations"] = 20
        config["warmup_iterations"] = 5
    
    if args.iterations != BENCHMARK_CONFIG["iterations"]:
        config["iterations"] = args.iterations
    
    # Run benchmarks
    runner = BenchmarkRunner(config)
    results = runner.run_all_benchmarks()
    
    # Save and display results
    runner.save_results(args.output)
    runner.print_summary()
    
    print(f"\n✅ Benchmarks completed successfully!")
    print(f"📁 Detailed results: {args.output}")


if __name__ == "__main__":
    main()