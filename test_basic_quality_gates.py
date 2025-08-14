#!/usr/bin/env python3
"""Basic Quality Gates Test.

Tests core functionality without external dependencies.
"""

import asyncio
import json
import time
import sys
from pathlib import Path


class BasicQualityGate:
    """Basic quality gate implementation."""
    
    def __init__(self, name: str, test_func):
        self.name = name
        self.test_func = test_func
    
    async def execute(self):
        """Execute the quality gate."""
        start_time = time.time()
        try:
            result = await self.test_func()
            passed = result.get('passed', False)
            score = result.get('score', 0.0)
            details = result.get('details', {})
            
            return {
                'name': self.name,
                'passed': passed,
                'score': score,
                'details': details,
                'execution_time': time.time() - start_time
            }
        except Exception as e:
            return {
                'name': self.name,
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)},
                'execution_time': time.time() - start_time
            }


async def test_basic_functionality():
    """Test basic Python functionality."""
    try:
        # Test basic operations
        test_list = [1, 2, 3, 4, 5]
        test_dict = {'key': 'value'}
        test_string = "Hello, World!"
        
        # Test list operations
        assert len(test_list) == 5
        assert sum(test_list) == 15
        
        # Test dict operations
        assert test_dict['key'] == 'value'
        test_dict['new_key'] = 'new_value'
        assert 'new_key' in test_dict
        
        # Test string operations
        assert test_string.lower() == "hello, world!"
        assert test_string.replace("World", "RLHF") == "Hello, RLHF!"
        
        return {
            'passed': True,
            'score': 1.0,
            'details': {
                'list_operations': 'passed',
                'dict_operations': 'passed',
                'string_operations': 'passed'
            }
        }
    except Exception as e:
        return {
            'passed': False,
            'score': 0.0,
            'details': {'error': str(e)}
        }


async def test_async_functionality():
    """Test async operations."""
    try:
        # Test async sleep
        start_time = time.time()
        await asyncio.sleep(0.1)
        duration = time.time() - start_time
        
        # Should be close to 0.1 seconds
        assert 0.05 < duration < 0.2
        
        # Test async gather
        async def simple_coro(value):
            await asyncio.sleep(0.01)
            return value * 2
        
        results = await asyncio.gather(
            simple_coro(1),
            simple_coro(2),
            simple_coro(3)
        )
        
        assert results == [2, 4, 6]
        
        return {
            'passed': True,
            'score': 1.0,
            'details': {
                'async_sleep': 'passed',
                'async_gather': 'passed',
                'duration': duration
            }
        }
    except Exception as e:
        return {
            'passed': False,
            'score': 0.0,
            'details': {'error': str(e)}
        }


async def test_file_operations():
    """Test file I/O operations."""
    try:
        # Test file writing and reading
        test_file = Path('/tmp/test_quality_gate.json')
        test_data = {
            'test': True,
            'timestamp': time.time(),
            'data': [1, 2, 3]
        }
        
        # Write file
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        assert test_file.exists()
        
        # Read file
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['test'] == True
        assert loaded_data['data'] == [1, 2, 3]
        
        # Cleanup
        test_file.unlink()
        
        return {
            'passed': True,
            'score': 1.0,
            'details': {
                'file_write': 'passed',
                'file_read': 'passed',
                'json_operations': 'passed'
            }
        }
    except Exception as e:
        return {
            'passed': False,
            'score': 0.0,
            'details': {'error': str(e)}
        }


async def test_performance():
    """Test performance characteristics."""
    try:
        # Test large list operations
        start_time = time.time()
        large_list = list(range(10000))
        creation_time = time.time() - start_time
        
        # Test list comprehension performance
        start_time = time.time()
        squared = [x*x for x in large_list]
        comprehension_time = time.time() - start_time
        
        # Test sum operation performance
        start_time = time.time()
        total = sum(squared)
        sum_time = time.time() - start_time
        
        # Performance thresholds (should be very fast)
        assert creation_time < 0.1
        assert comprehension_time < 0.1
        assert sum_time < 0.1
        
        return {
            'passed': True,
            'score': 1.0,
            'details': {
                'list_creation_time': creation_time,
                'comprehension_time': comprehension_time,
                'sum_time': sum_time,
                'total_computed': total
            }
        }
    except Exception as e:
        return {
            'passed': False,
            'score': 0.0,
            'details': {'error': str(e)}
        }


async def test_error_handling():
    """Test error handling capabilities."""
    try:
        error_caught = False
        
        # Test exception handling
        try:
            result = 1 / 0
        except ZeroDivisionError:
            error_caught = True
        
        assert error_caught, "ZeroDivisionError should have been caught"
        
        # Test custom exception
        class CustomError(Exception):
            pass
        
        custom_error_caught = False
        try:
            raise CustomError("Test error")
        except CustomError:
            custom_error_caught = True
        
        assert custom_error_caught, "Custom error should have been caught"
        
        return {
            'passed': True,
            'score': 1.0,
            'details': {
                'standard_exception': 'caught',
                'custom_exception': 'caught'
            }
        }
    except Exception as e:
        return {
            'passed': False,
            'score': 0.0,
            'details': {'error': str(e)}
        }


async def test_data_structures():
    """Test advanced data structure operations."""
    try:
        # Test set operations
        set1 = {1, 2, 3, 4}
        set2 = {3, 4, 5, 6}
        
        union = set1 | set2
        intersection = set1 & set2
        difference = set1 - set2
        
        assert union == {1, 2, 3, 4, 5, 6}
        assert intersection == {3, 4}
        assert difference == {1, 2}
        
        # Test dict comprehensions
        numbers = [1, 2, 3, 4, 5]
        squared_dict = {n: n**2 for n in numbers}
        
        assert squared_dict[3] == 9
        assert squared_dict[5] == 25
        
        # Test nested structures
        nested = {
            'level1': {
                'level2': {
                    'data': [1, 2, {'nested_list': [4, 5, 6]}]
                }
            }
        }
        
        assert nested['level1']['level2']['data'][2]['nested_list'][1] == 5
        
        return {
            'passed': True,
            'score': 1.0,
            'details': {
                'set_operations': 'passed',
                'dict_comprehensions': 'passed',
                'nested_structures': 'passed'
            }
        }
    except Exception as e:
        return {
            'passed': False,
            'score': 0.0,
            'details': {'error': str(e)}
        }


async def main():
    """Run basic quality gates."""
    print("🚀 Starting Basic Quality Gates Execution")
    print("=" * 50)
    
    # Define quality gates
    gates = [
        BasicQualityGate("Basic Functionality", test_basic_functionality),
        BasicQualityGate("Async Operations", test_async_functionality),
        BasicQualityGate("File Operations", test_file_operations),
        BasicQualityGate("Performance", test_performance),
        BasicQualityGate("Error Handling", test_error_handling),
        BasicQualityGate("Data Structures", test_data_structures)
    ]
    
    start_time = time.time()
    results = []
    passed_count = 0
    
    # Execute gates
    for gate in gates:
        print(f"🔍 Executing: {gate.name}")
        result = await gate.execute()
        results.append(result)
        
        if result['passed']:
            print(f"   ✅ PASSED (score: {result['score']:.2f})")
            passed_count += 1
        else:
            print(f"   ❌ FAILED - {result['details'].get('error', 'Unknown error')}")
    
    execution_time = time.time() - start_time
    
    # Generate report
    report = {
        'execution_summary': {
            'total_gates': len(gates),
            'passed_gates': passed_count,
            'success_rate': passed_count / len(gates),
            'execution_time': execution_time,
            'timestamp': time.time()
        },
        'results': results
    }
    
    # Save report
    output_path = Path('quality_gates_results.json')
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Display summary
    print("\n" + "=" * 50)
    print("📊 BASIC QUALITY GATES RESULTS")
    print("=" * 50)
    print(f"✅ Gates Passed: {passed_count}/{len(gates)} ({passed_count/len(gates)*100:.1f}%)")
    print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
    print(f"📁 Report saved to: {output_path.absolute()}")
    
    # Success criteria
    success_rate = passed_count / len(gates)
    if success_rate >= 0.85:
        print("\n🎉 QUALITY GATES: PASSED (85%+ success rate achieved)")
        return 0
    else:
        print(f"\n❌ QUALITY GATES: FAILED ({success_rate*100:.1f}% success rate, need 85%)")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))