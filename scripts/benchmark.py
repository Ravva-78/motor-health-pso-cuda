import time
import requests
import statistics
import sys
import os
import numpy as np

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

def benchmark_direct_engine(num_iterations=200):
    """Fallback benchmark directly testing ONNX Inference Engine when API is offline."""
    print("REST API offline — executing Direct In-Memory ONNX Inference Engine benchmark...")
    from backend.inference_service import InferenceService
    
    engine = InferenceService()
    dummy_features = [0.45, 0.52, 0.48, 0.55, 0.30]
    
    # Warm-up
    for _ in range(20):
        engine.predict(dummy_features)
        
    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        res = engine.predict(dummy_features)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
    return {
        "count": len(latencies),
        "errors": 0,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "provider": str(engine.session.get_providers()[0] if engine.session else "Unknown")
    }

if __name__ == "__main__":
    print("AeroForge Performance Benchmark")
    print("===================================")
    
    # Try connecting to API
    api_online = False
    try:
        r = requests.get(f"{API_URL}/health", timeout=1)
        if r.status_code == 200:
            api_online = True
    except Exception:
        pass
        
    if api_online:
        print(f"Connected to REST API at {API_URL}.")
        print("Measuring REST API -> IPC -> Daemon /predict latency (100 sequential requests)...")
        res = measure_predict_latency(100)
    else:
        print(f"Notice: REST API not running at {API_URL}.")
        res = benchmark_direct_engine(200)
    
    if res:
        print(f"Successful Runs:     {res['count']}")
        print(f"Failed Runs:         {res['errors']}")
        print(f"Mean Latency:        {res['mean_ms']:.3f} ms")
        print(f"Median Latency:      {res['median_ms']:.3f} ms")
        print(f"Min Latency:         {res['min_ms']:.3f} ms")
        print(f"Max Latency:         {res['max_ms']:.3f} ms")
        throughput = 1000.0 / res['mean_ms'] if res['mean_ms'] > 0 else 0
        print(f"Approx. Throughput:  {throughput:.1f} requests/sec (Single Thread)")
        if "provider" in res:
            print(f"Active Provider:     {res['provider']}")
    else:
        print("All benchmark runs failed.")

