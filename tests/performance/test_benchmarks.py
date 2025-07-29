"""Performance benchmark tests using pytest-benchmark.

These tests measure the performance characteristics of core RLHF Audit Trail
operations to detect performance regressions and establish baselines.
"""

import pytest
import json
import time
import hashlib
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List


class TestAuditLoggingPerformance:
    """Benchmark audit logging operations."""
    
    @pytest.fixture
    def sample_audit_data(self) -> Dict[str, Any]:
        """Sample audit data for benchmarking."""
        return {
            "event_type": "annotation",
            "timestamp": time.time(),
            "event_data": {
                "prompt_hash": "sha256:abcd1234" + "x" * 56,
                "response_hash": "sha256:efgh5678" + "y" * 56,
                "annotator_id": "dp_anonymized_id_001",
                "reward": 0.85,
                "privacy_noise": 0.02
            },
            "policy_state": {
                "checkpoint": "epoch_5_step_1000",
                "parameter_delta_norm": 0.015,
                "gradient_stats": {"mean": 0.01, "std": 0.05}
            },
            "metadata": {
                "session_id": "session_123",
                "experiment_id": "exp_456",
                "model_version": "v1.2.3"
            }
        }
    
    @pytest.mark.benchmark
    def test_json_serialization_performance(self, benchmark, sample_audit_data):
        """Benchmark JSON serialization of audit data."""
        result = benchmark(json.dumps, sample_audit_data)
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.benchmark
    def test_hash_computation_performance(self, benchmark, sample_audit_data):
        """Benchmark hash computation for audit integrity."""
        def compute_hash(data):
            serialized = json.dumps(data, sort_keys=True)
            return hashlib.sha256(serialized.encode()).hexdigest()
        
        result = benchmark(compute_hash, sample_audit_data)
        assert len(result) == 64  # SHA-256 hex length
    
    @pytest.mark.benchmark
    def test_audit_record_creation(self, benchmark):
        """Benchmark creation of audit records."""
        def create_audit_record(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "id": f"audit_{int(time.time() * 1000000)}",
                "timestamp": time.time(),
                "event_type": event_type,
                "data": data,
                "integrity_hash": hashlib.sha256(
                    json.dumps(data, sort_keys=True).encode()
                ).hexdigest()
            }
        
        test_data = {"key": "value", "number": 42}
        result = benchmark(create_audit_record, "test_event", test_data)
        assert result["event_type"] == "test_event"
        assert "integrity_hash" in result


class TestPrivacyComputationPerformance:
    """Benchmark privacy computation operations."""
    
    @pytest.mark.benchmark
    def test_differential_privacy_noise_generation(self, benchmark):
        """Benchmark differential privacy noise generation.""" 
        def generate_dp_noise(epsilon: float, sensitivity: float, size: int = 100) -> List[float]:
            import random
            return [random.gauss(0, sensitivity / epsilon) for _ in range(size)]
        
        result = benchmark(generate_dp_noise, 1.0, 1.0, 100)
        assert len(result) == 100
        assert all(isinstance(x, float) for x in result)
    
    @pytest.mark.benchmark
    def test_privacy_budget_calculation(self, benchmark):
        """Benchmark privacy budget calculations."""
        def calculate_privacy_budget(operations: List[float], total_budget: float) -> Dict[str, float]:
            used_budget = sum(operations)
            return {
                "used": used_budget,
                "remaining": total_budget - used_budget,
                "utilization": used_budget / total_budget if total_budget > 0 else 0
            }
        
        operations = [0.1] * 50  # 50 operations with epsilon=0.1 each
        result = benchmark(calculate_privacy_budget, operations, 10.0)
        assert result["used"] == 5.0
        assert result["remaining"] == 5.0
        assert result["utilization"] == 0.5


class TestModelOperationPerformance:
    """Benchmark model-related operations."""
    
    @pytest.fixture
    def mock_model_weights(self) -> Dict[str, List[float]]:
        """Mock model weights for benchmarking."""
        return {
            f"layer_{i}": [0.1] * 1000 for i in range(10)
        }
    
    @pytest.mark.benchmark
    def test_model_checkpoint_serialization(self, benchmark, mock_model_weights):
        """Benchmark model checkpoint serialization."""
        checkpoint_data = {
            "model_state_dict": mock_model_weights,
            "optimizer_state": {"lr": 0.001, "momentum": 0.9},
            "epoch": 10,
            "step": 1000,
            "metrics": {"loss": 0.5, "accuracy": 0.85}
        }
        
        result = benchmark(json.dumps, checkpoint_data)
        assert isinstance(result, str)
        assert len(result) > 1000  # Should be substantial
    
    @pytest.mark.benchmark
    def test_parameter_diff_calculation(self, benchmark, mock_model_weights):
        """Benchmark calculation of parameter differences."""
        def calculate_param_diff(weights_before: Dict[str, List[float]], 
                               weights_after: Dict[str, List[float]]) -> Dict[str, float]:
            diffs = {}
            for layer_name in weights_before:
                if layer_name in weights_after:
                    before = weights_before[layer_name]
                    after = weights_after[layer_name]
                    diff = sum((a - b) ** 2 for a, b in zip(after, before)) ** 0.5
                    diffs[layer_name] = diff
            return diffs
        
        # Create slightly modified weights
        modified_weights = {
            layer: [w + 0.01 for w in weights] 
            for layer, weights in mock_model_weights.items()
        }
        
        result = benchmark(calculate_param_diff, mock_model_weights, modified_weights)
        assert len(result) == len(mock_model_weights)
        assert all(diff > 0 for diff in result.values())


class TestDatabaseOperationPerformance:
    """Benchmark database operation simulations."""
    
    @pytest.fixture
    def mock_database(self) -> Dict[str, Any]:
        """Mock database for benchmarking."""
        return {}
    
    @pytest.mark.benchmark
    def test_bulk_insert_performance(self, benchmark, mock_database):
        """Benchmark bulk insert operations."""
        def bulk_insert(db: Dict[str, Any], records: List[Dict[str, Any]]) -> int:
            count = 0
            for record in records:
                db[record["id"]] = record
                count += 1
            return count
        
        records = [
            {"id": f"record_{i}", "data": f"content_{i}", "timestamp": time.time()}
            for i in range(100)
        ]
        
        result = benchmark(bulk_insert, mock_database, records)
        assert result == 100
        assert len(mock_database) == 100
    
    @pytest.mark.benchmark
    def test_query_performance(self, benchmark, mock_database):
        """Benchmark query operations."""
        # Pre-populate database
        for i in range(1000):
            mock_database[f"record_{i}"] = {
                "id": f"record_{i}",
                "data": f"content_{i}",
                "category": f"cat_{i % 10}"
            }
        
        def query_by_category(db: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
            return [
                record for record in db.values()
                if record.get("category") == category
            ]
        
        result = benchmark(query_by_category, mock_database, "cat_5")
        assert len(result) == 100  # Should find 100 records with category "cat_5"


class TestMemoryUsagePerformance:
    """Memory usage benchmarks."""
    
    @pytest.mark.benchmark
    def test_large_audit_log_memory_usage(self, benchmark):
        """Benchmark memory usage for large audit logs."""
        def create_large_audit_log(num_entries: int) -> List[Dict[str, Any]]:
            return [
                {
                    "id": i,
                    "timestamp": time.time(),
                    "event_type": "annotation",
                    "data": {
                        "prompt": "x" * 1000,  # 1KB prompt
                        "response": "y" * 1000,  # 1KB response
                        "metadata": {"key": "value"} * 10
                    }
                }
                for i in range(num_entries)
            ]
        
        result = benchmark(create_large_audit_log, 100)
        assert len(result) == 100
        assert all("data" in entry for entry in result)


class TestConcurrencyPerformance:
    """Concurrency and thread safety benchmarks."""
    
    @pytest.mark.benchmark
    def test_concurrent_audit_logging(self, benchmark):
        """Benchmark concurrent audit logging simulation."""
        import threading
        import queue
        
        def concurrent_audit_logging(num_threads: int, operations_per_thread: int) -> Dict[str, Any]:
            results_queue = queue.Queue()
            threads = []
            
            def worker():
                local_results = []
                for i in range(operations_per_thread):
                    audit_record = {
                        "id": f"audit_{threading.current_thread().ident}_{i}",
                        "timestamp": time.time(),
                        "data": {"operation": i}
                    }
                    # Simulate audit logging work
                    json.dumps(audit_record)
                    local_results.append(audit_record)
                results_queue.put(local_results)
            
            # Start threads
            for _ in range(num_threads):
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Collect results
            all_results = []
            while not results_queue.empty():
                all_results.extend(results_queue.get())
            
            return {
                "total_operations": len(all_results),
                "threads_used": num_threads,
                "operations_per_thread": operations_per_thread
            }
        
        result = benchmark(concurrent_audit_logging, 4, 25)  # 4 threads, 25 ops each
        assert result["total_operations"] == 100
        assert result["threads_used"] == 4


# Performance regression tests
class TestPerformanceRegression:
    """Tests to detect performance regressions."""
    
    @pytest.mark.benchmark
    def test_audit_logging_regression(self, benchmark):
        """Ensure audit logging performance doesn't regress."""
        def audit_log_operation():
            data = {"event": "test", "timestamp": time.time()}
            serialized = json.dumps(data)
            hashed = hashlib.sha256(serialized.encode()).hexdigest()
            return {"serialized": serialized, "hash": hashed}
        
        result = benchmark(audit_log_operation)
        assert len(result["hash"]) == 64
        
        # Performance expectations (adjust based on baseline measurements)
        # These would typically be set after establishing baselines
        assert benchmark.stats.stats.mean < 0.01  # Less than 10ms on average
    
    @pytest.mark.benchmark
    def test_privacy_computation_regression(self, benchmark):
        """Ensure privacy computation performance doesn't regress."""
        def privacy_operation():
            import random
            # Simulate differential privacy computation
            noise_values = [random.gauss(0, 1.0) for _ in range(100)]
            return sum(noise_values) / len(noise_values)
        
        result = benchmark(privacy_operation)
        assert isinstance(result, float)
        
        # Performance expectation
        assert benchmark.stats.stats.mean < 0.005  # Less than 5ms on average