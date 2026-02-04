#!/usr/bin/env python3
"""
Latency test script for GGUF backend
Run this to test for latency spikes and monitor performance
"""

import time
import statistics
import threading
from typing import List, Dict
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backends.gguf_backend import GGUFClient

def test_latency_spikes(model_path: str, num_requests: int = 20, concurrent: bool = False):
    """Test for latency spikes by running multiple requests"""
    
    print(f"🚀 Starting latency test with {num_requests} requests")
    print(f"📊 Model: {model_path}")
    print(f"🔧 Concurrent: {concurrent}")
    
    # Initialize the client
    client = GGUFClient(model_path=model_path, chat_format="chatml")
    
    # Test messages - simple queries that should be fast
    test_messages = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Hello, how are you?"}],
        [{"role": "user", "content": "What's the weather like?"}],
        [{"role": "user", "content": "Tell me a joke"}],
        [{"role": "user", "content": "What time is it?"}],
    ]
    
    latencies = []
    errors = []
    
    def run_single_request(request_id: int):
        """Run a single request and record timing"""
        start_time = time.time()
        try:
            # Use a simple message for consistent testing
            messages = test_messages[request_id % len(test_messages)]
            response = client.chat_with_temperature(messages, temperature=0.0)
            end_time = time.time()
            latency = end_time - start_time
            
            latencies.append(latency)
            print(f"✅ Request {request_id+1}: {latency:.3f}s")
            
            # Check for latency spikes
            if latency > 5.0:  # More than 5 seconds is a spike
                print(f"⚠️  LATENCY SPIKE DETECTED: {latency:.3f}s for request {request_id+1}")
                
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            errors.append((request_id, str(e), latency))
            print(f"❌ Request {request_id+1} failed after {latency:.3f}s: {e}")
    
    if concurrent:
        # Run requests concurrently
        threads = []
        for i in range(num_requests):
            thread = threading.Thread(target=run_single_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
    else:
        # Run requests sequentially
        for i in range(num_requests):
            run_single_request(i)
    
    # Calculate statistics
    if latencies:
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        print(f"\n📊 LATENCY STATISTICS:")
        print(f"   Average: {avg_latency:.3f}s")
        print(f"   Median:  {median_latency:.3f}s")
        print(f"   Min:     {min_latency:.3f}s")
        print(f"   Max:     {max_latency:.3f}s")
        print(f"   Std Dev: {std_dev:.3f}s")
        
        # Identify spikes (requests that are 2x the median)
        spikes = [lat for lat in latencies if lat > median_latency * 2]
        if spikes:
            print(f"⚠️  Found {len(spikes)} latency spikes (>2x median):")
            for spike in spikes:
                print(f"   - {spike:.3f}s")
        else:
            print(f"✅ No significant latency spikes detected")
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for request_id, error, latency in errors:
            print(f"   Request {request_id+1}: {error} (after {latency:.3f}s)")
    
    # Print cache statistics if available
    try:
        cache_stats = client.get_cache_stats()
        print(f"\n💾 CACHE STATISTICS:")
        print(f"   Hits: {cache_stats['cache_hits']}")
        print(f"   Misses: {cache_stats['cache_misses']}")
        print(f"   Hit Rate: {cache_stats['hit_rate']:.1f}%")
        print(f"   Cache Size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
    except (AttributeError, KeyError, TypeError):
        pass
    
    # Cleanup
    client.unload()
    
    return {
        "latencies": latencies,
        "errors": errors,
        "avg_latency": avg_latency if latencies else 0,
        "max_latency": max_latency if latencies else 0,
        "spikes": len(spikes) if latencies else 0
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GGUF backend for latency spikes")
    parser.add_argument("model_path", help="Path to the GGUF model file")
    parser.add_argument("--requests", "-r", type=int, default=20, help="Number of requests to test")
    parser.add_argument("--concurrent", "-c", action="store_true", help="Run requests concurrently")
    parser.add_argument("--iterations", "-i", type=int, default=1, help="Number of test iterations")
    
    args = parser.parse_args()
    
    print("🔍 GGUF Backend Latency Test")
    print("=" * 50)
    
    all_results = []
    
    for iteration in range(args.iterations):
        print(f"\n🔄 ITERATION {iteration + 1}/{args.iterations}")
        print("-" * 30)
        
        result = test_latency_spikes(
            model_path=args.model_path,
            num_requests=args.requests,
            concurrent=args.concurrent
        )
        all_results.append(result)
        
        if args.iterations > 1:
            time.sleep(2)  # Brief pause between iterations
    
    # Summary across all iterations
    if args.iterations > 1:
        print(f"\n📈 SUMMARY ACROSS {args.iterations} ITERATIONS:")
        print("=" * 50)
        
        avg_latencies = [r["avg_latency"] for r in all_results if r["avg_latency"] > 0]
        max_latencies = [r["max_latency"] for r in all_results if r["max_latency"] > 0]
        total_spikes = sum(r["spikes"] for r in all_results)
        total_errors = sum(len(r["errors"]) for r in all_results)
        
        if avg_latencies:
            print(f"   Average Latency: {statistics.mean(avg_latencies):.3f}s")
            print(f"   Best Average: {min(avg_latencies):.3f}s")
            print(f"   Worst Average: {max(avg_latencies):.3f}s")
        
        if max_latencies:
            print(f"   Average Max Latency: {statistics.mean(max_latencies):.3f}s")
            print(f"   Overall Max Latency: {max(max_latencies):.3f}s")
        
        print(f"   Total Spikes: {total_spikes}")
        print(f"   Total Errors: {total_errors}")
    
    print(f"\n✅ Test completed!")
