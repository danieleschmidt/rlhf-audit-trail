#!/usr/bin/env python3
"""
Advanced Performance Monitoring Script for RLHF Audit Trail
High-maturity repository performance tracking and regression detection
"""

import json
import logging
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Advanced performance monitoring for Python AI/ML applications."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.project_root = Path(__file__).parent.parent
        self.config_path = config_path or self.project_root / "monitoring" / "performance" / "performance-monitoring.yml"
        self.results_dir = self.project_root / "benchmarks" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
        self.baseline = self._load_baseline()
        
    def _load_config(self) -> Dict:
        """Load performance monitoring configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file is missing."""
        return {
            "benchmarks": {
                "core_functionality": [
                    {
                        "name": "basic_performance",
                        "target_ops_per_second": 100,
                        "max_latency_ms": 100,
                        "memory_limit_mb": 500
                    }
                ]
            },
            "regression_detection": {
                "enabled": True,
                "threshold_percentage": 10
            }
        }
        
    def _load_baseline(self) -> Optional[Dict]:
        """Load baseline performance metrics."""
        baseline_file = self.project_root / "benchmarks" / "baseline_performance.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load baseline: {e}")
        return None
        
    def measure_system_performance(self) -> Dict:
        """Measure current system performance metrics."""
        logger.info("Measuring system performance...")
        
        # CPU and Memory metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu = process.cpu_percent()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_percent": memory.percent,
                "disk_total_gb": disk.total / (1024**3),
                "disk_used_gb": disk.used / (1024**3),
                "disk_percent": (disk.used / disk.total) * 100
            },
            "process": {
                "memory_rss_mb": process_memory.rss / (1024**2),
                "memory_vms_mb": process_memory.vms / (1024**2),
                "cpu_percent": process_cpu
            }
        }
        
    def run_benchmark(self, benchmark_config: Dict) -> Dict:
        """Run a specific benchmark and measure performance."""
        benchmark_name = benchmark_config["name"]
        logger.info(f"Running benchmark: {benchmark_name}")
        
        # Simulate benchmark execution
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Placeholder for actual benchmark logic
        # In a real implementation, this would run the specific benchmark
        operations_count = 0
        latencies = []
        
        # Simulate operations
        target_ops = benchmark_config.get("target_ops_per_second", 100)
        duration_seconds = 10  # Run for 10 seconds
        
        end_time = start_time + duration_seconds
        while time.time() < end_time:
            op_start = time.time()
            # Simulate work
            time.sleep(0.001)  # 1ms simulated work
            op_end = time.time()
            
            latencies.append((op_end - op_start) * 1000)  # Convert to milliseconds
            operations_count += 1
            
        total_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss
        
        # Calculate metrics
        ops_per_second = operations_count / total_time
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18] if latencies else 0  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98] if latencies else 0  # 99th percentile
        memory_used_mb = (end_memory - start_memory) / (1024**2)
        
        return {
            "benchmark_name": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "operations_per_second": ops_per_second,
                "total_operations": operations_count,
                "average_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "p99_latency_ms": p99_latency,
                "memory_used_mb": memory_used_mb,
                "duration_seconds": total_time
            },
            "targets": {
                "target_ops_per_second": benchmark_config.get("target_ops_per_second", 0),
                "max_latency_ms": benchmark_config.get("max_latency_ms", 0),
                "memory_limit_mb": benchmark_config.get("memory_limit_mb", 0)
            }
        }
        
    def detect_regression(self, current_results: Dict, baseline: Optional[Dict] = None) -> Dict:
        """Detect performance regression compared to baseline."""
        if not baseline:
            baseline = self.baseline
            
        if not baseline:
            logger.warning("No baseline available for regression detection")
            return {"regression_detected": False, "reason": "No baseline available"}
            
        regression_config = self.config.get("regression_detection", {})
        threshold = regression_config.get("threshold_percentage", 10)
        
        regressions = []
        improvements = []
        
        for benchmark_name, current_metrics in current_results.items():
            if benchmark_name not in baseline:
                continue
                
            baseline_metrics = baseline[benchmark_name]["metrics"]
            current_metric_values = current_metrics["metrics"]
            
            # Check key performance indicators
            checks = [
                ("operations_per_second", "higher_is_better"),
                ("average_latency_ms", "lower_is_better"),
                ("p95_latency_ms", "lower_is_better"),
                ("memory_used_mb", "lower_is_better")
            ]
            
            for metric_name, direction in checks:
                if metric_name not in baseline_metrics or metric_name not in current_metric_values:
                    continue
                    
                baseline_value = baseline_metrics[metric_name]
                current_value = current_metric_values[metric_name]
                
                if baseline_value == 0:
                    continue
                    
                percentage_change = ((current_value - baseline_value) / baseline_value) * 100
                
                if direction == "higher_is_better":
                    if percentage_change < -threshold:
                        regressions.append({
                            "benchmark": benchmark_name,
                            "metric": metric_name,
                            "baseline_value": baseline_value,
                            "current_value": current_value,
                            "percentage_change": percentage_change
                        })
                    elif percentage_change > threshold:
                        improvements.append({
                            "benchmark": benchmark_name,
                            "metric": metric_name,
                            "baseline_value": baseline_value,
                            "current_value": current_value,
                            "percentage_change": percentage_change
                        })
                else:  # lower_is_better
                    if percentage_change > threshold:
                        regressions.append({
                            "benchmark": benchmark_name,
                            "metric": metric_name,
                            "baseline_value": baseline_value,
                            "current_value": current_value,
                            "percentage_change": percentage_change
                        })
                    elif percentage_change < -threshold:
                        improvements.append({
                            "benchmark": benchmark_name,
                            "metric": metric_name,
                            "baseline_value": baseline_value,
                            "current_value": current_value,
                            "percentage_change": percentage_change
                        })
        
        return {
            "regression_detected": len(regressions) > 0,
            "regressions": regressions,
            "improvements": improvements,
            "threshold_percentage": threshold
        }
        
    def run_all_benchmarks(self) -> Dict:
        """Run all configured benchmarks."""
        logger.info("Running all performance benchmarks...")
        
        all_results = {}
        system_metrics = self.measure_system_performance()
        
        # Run core functionality benchmarks
        core_benchmarks = self.config.get("benchmarks", {}).get("core_functionality", [])
        for benchmark_config in core_benchmarks:
            try:
                result = self.run_benchmark(benchmark_config)
                all_results[result["benchmark_name"]] = result
            except Exception as e:
                logger.error(f"Benchmark failed: {benchmark_config['name']}: {e}")
                
        # Run ML operation benchmarks if configured
        ml_benchmarks = self.config.get("benchmarks", {}).get("ml_operations", [])
        for benchmark_config in ml_benchmarks:
            try:
                result = self.run_benchmark(benchmark_config)
                all_results[result["benchmark_name"]] = result
            except Exception as e:
                logger.error(f"ML benchmark failed: {benchmark_config['name']}: {e}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"performance_results_{timestamp}.json"
        
        full_results = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": system_metrics,
            "benchmark_results": all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
            
        logger.info(f"Performance results saved: {results_file}")
        
        # Check for regressions
        regression_analysis = self.detect_regression(all_results)
        if regression_analysis["regression_detected"]:
            logger.warning(f"Performance regression detected: {len(regression_analysis['regressions'])} issues")
            for regression in regression_analysis['regressions']:
                logger.warning(
                    f"  {regression['benchmark']}.{regression['metric']}: "
                    f"{regression['percentage_change']:.1f}% change "
                    f"({regression['baseline_value']} -> {regression['current_value']})"
                )
        else:
            logger.info("No performance regressions detected")
            
        if regression_analysis["improvements"]:
            logger.info(f"Performance improvements detected: {len(regression_analysis['improvements'])} metrics")
            
        return {
            "results": full_results,
            "regression_analysis": regression_analysis
        }
        
    def update_baseline(self, results: Dict):
        """Update baseline performance metrics."""
        baseline_file = self.project_root / "benchmarks" / "baseline_performance.json"
        
        # Extract just the benchmark results for baseline
        baseline_data = {}
        for benchmark_name, result in results["benchmark_results"].items():
            baseline_data[benchmark_name] = {
                "timestamp": result["timestamp"],
                "metrics": result["metrics"]
            }
            
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
            
        logger.info(f"Baseline updated: {baseline_file}")


def main():
    """Main entry point for performance monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Performance Monitor")
    parser.add_argument("--update-baseline", action="store_true",
                       help="Update baseline metrics after running benchmarks")
    parser.add_argument("--config", type=Path,
                       help="Path to performance monitoring config file")
    
    args = parser.parse_args()
    
    try:
        monitor = PerformanceMonitor(config_path=args.config)
        results = monitor.run_all_benchmarks()
        
        if args.update_baseline:
            monitor.update_baseline(results["results"])
            
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE MONITORING SUMMARY")
        print("="*60)
        
        benchmark_results = results["results"]["benchmark_results"]
        print(f"Benchmarks run: {len(benchmark_results)}")
        
        for name, result in benchmark_results.items():
            metrics = result["metrics"]
            targets = result["targets"]
            
            print(f"\n{name}:")
            print(f"  Operations/sec: {metrics['operations_per_second']:.1f} "
                  f"(target: {targets.get('target_ops_per_second', 'N/A')})")
            print(f"  Avg latency: {metrics['average_latency_ms']:.2f}ms "
                  f"(max: {targets.get('max_latency_ms', 'N/A')}ms)")
            print(f"  Memory used: {metrics['memory_used_mb']:.1f}MB "
                  f"(limit: {targets.get('memory_limit_mb', 'N/A')}MB)")
                  
        regression_analysis = results["regression_analysis"]
        if regression_analysis["regression_detected"]:
            print(f"\n⚠️  Performance regressions detected: {len(regression_analysis['regressions'])}")
            return 1
        else:
            print(f"\n✅ No performance regressions detected")
            
        if regression_analysis["improvements"]:
            print(f"🚀 Performance improvements: {len(regression_analysis['improvements'])}")
            
        print("="*60)
        return 0
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())