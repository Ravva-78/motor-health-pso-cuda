import time
import requests
import statistics
import sys
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")

def measure_predict_latency(num_requests=100):
    latencies = []
    errors = 0
    
    for _ in range(num_requests):
        start = time.perf_counter()
        try:
            r = requests.post(f"{API_URL}/predict", timeout=2)
            end = time.perf_counter()
            if r.status_code == 200:
                latencies.append((end - start) * 1000) # Convert to ms
            else:
                errors += 1
        except Exception:
            errors += 1
            
    if not latencies:
        return None
        
    return {
        "count": len(latencies),
        "errors": errors,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies)
    }

if __name__ == "__main__":
    print("AeroForge E2E Performance Benchmark")
    print("===================================")
    
    # Wait for API to be ready
    print(f"Connecting to {API_URL}...")
    try:
        requests.get(f"{API_URL}/health", timeout=2)
    except Exception as e:
        print(f"API unreachable: {e}")
        sys.exit(1)
        
    print("Measuring REST API -> IPC -> Daemon /predict latency (100 sequential requests)...")
    res = measure_predict_latency(100)
    
    if res:
        print(f"Successful Requests: {res['count']}")
        print(f"Failed Requests:     {res['errors']}")
        print(f"Mean Latency:        {res['mean_ms']:.2f} ms")
        print(f"Median Latency:      {res['median_ms']:.2f} ms")
        print(f"Min Latency:         {res['min_ms']:.2f} ms")
        print(f"Max Latency:         {res['max_ms']:.2f} ms")
        print(f"Approx. Throughput:  {1000 / res['mean_ms']:.2f} requests/sec (Single Thread)")
    else:
        print("All requests failed.")
