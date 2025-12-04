import time
import torch
import numpy as np
from app.models.neural_net import PricePredictionModel

def run_benchmark():
    print("==================================================")
    print("⚡️ Switch AI Engine Performance Benchmark v1.2.0")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Acceleration: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = PricePredictionModel(input_dim=128).to(device)
    batch_sizes = [1, 16, 64, 256]
    
    print("\nInference Latency Tests:")
    print("--------------------------------------------------")
    print(f"{'Batch Size':<15} | {'Latency (ms)':<15} | {'Throughput (req/s)':<20}")
    print("--------------------------------------------------")
    
    for bs in batch_sizes:
        input_tensor = torch.randn(bs, 128).to(device)
        
        # Warmup
        for _ in range(10):
            _ = model(input_tensor)
            
        # Test
        start_time = time.time()
        iterations = 100
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_tensor)
        
        end_time = time.time()
        avg_latency = ((end_time - start_time) / iterations) * 1000
        throughput = (bs * iterations) / (end_time - start_time)
        
        print(f"{bs:<15} | {avg_latency:.2f}{' ms':<12} | {throughput:.0f}")

    print("\n✅ Benchmark Complete. System ready for high-frequency trading analysis.")

if __name__ == "__main__":
    run_benchmark()
