#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from ensemble_attribution import shapley_exact_or_fast
import torch
import torch.nn as nn

# Create dummy models
class DummyModel(nn.Module):
    def __init__(self, name, bias=0.0):
        super().__init__()
        self.name = name
        self.linear = nn.Linear(10, 5)
        self.bias = bias
    def forward(self, x):
        return self.linear(x.flatten(1)) + self.bias

if __name__ == "__main__":
    # Test ensemble
    models = [DummyModel(f'Model_{i}', bias=i*0.1) for i in range(5)]
    iq_tensor = torch.randn(1, 10)

    # Test exact Shapley
    import time
    t0 = time.perf_counter()
    shap = shapley_exact_or_fast(models, iq_tensor, exact_max_members=8)
    t1 = time.perf_counter()

    print('🎯 Exact Shapley Results:')
    for name, contrib in sorted(shap.items(), key=lambda x: x[1], reverse=True):
        print(f'  {name}: {contrib:.6f}')
    print(f'⏱️  Timing: {(t1-t0)*1000:.3f} ms')
    print(f'📊 Sum of contributions: {sum(shap.values()):.6f}')
    
    # Test with larger ensemble (should use MC approximation)
    print('\n🔄 Testing MC approximation with 10 models:')
    large_models = [DummyModel(f'LargeModel_{i}', bias=i*0.05) for i in range(10)]
    t0 = time.perf_counter()
    shap_large = shapley_exact_or_fast(large_models, iq_tensor, exact_max_members=8)
    t1 = time.perf_counter()
    
    print(f'📊 Top 3 contributors:')
    for name, contrib in sorted(shap_large.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f'  {name}: {contrib:.6f}')
    print(f'⏱️  Timing: {(t1-t0)*1000:.3f} ms')
    print(f'📊 Sum of contributions: {sum(shap_large.values()):.6f}')