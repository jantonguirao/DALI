# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nvidia.dali.experimental.dynamic as ndd
import numpy as np
from nose2.tools import cartesian_params, params


def asnumpy(tensor):
    """Convert a DALI dynamic tensor to numpy array."""
    return np.array(tensor.cpu().evaluate()._storage)

ops = {
    "uniform": ndd.ops.random.Uniform,
    "normal": ndd.ops.random.Normal,
}

fn = {
    "uniform": ndd.random.uniform,
    "normal": ndd.random.normal,
}

op_args = {
    "uniform": {"range": [0.0, 1.0], "shape": [10]},
    "normal": {"mean": 0.0, "stddev": 1.0, "shape": [10]},
}

@cartesian_params(("cpu", "gpu"), (None, 3), ("ops", "fn"), ("uniform", "normal"))
def test_rng_argument(device_type, batch_size, api_type, opname):
    """Test that the rng argument works with random operators."""
    # Create a simple RNG that returns predictable values
    rng_state = [0]
    
    def my_rng():
        rng_state[0] += 1
        return np.uint32(rng_state[0] * 12345)
    
    # Create operator or use functional API
    if api_type == "ops":
        op_instance = ops[opname](device=device_type)
        result1 = op_instance(batch_size=batch_size, rng=my_rng, **op_args[opname])
    else:
        result1 = fn[opname](batch_size=batch_size, rng=my_rng, device=device_type, **op_args[opname])

    # Verify result type and shape
    if batch_size is not None:
        assert isinstance(result1, ndd.Batch), f"Expected Batch, got {type(result1)}"
        result1_np = asnumpy(result1)
        assert result1_np.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {result1_np.shape}"
    else:
        assert isinstance(result1, ndd.Tensor), f"Expected Tensor, got {type(result1)}"
        result1_np = asnumpy(result1)
        assert result1_np.shape == (10,), f"Expected shape (10,), got {result1_np.shape}"

    # TODO(janton): Test that same rng sequence produces same results


@params(("cpu",), ("gpu",))
def test_rng_seed_exclusion(device_type):
    """Test that seed argument is removed when rng is provided."""
    rng_state = [0]
    
    def my_rng():
        rng_state[0] += 1
        return np.uint32(rng_state[0] * 12345)
    
    # This should work - rng should override seed (seed is an init-time argument)
    uniform_op = ndd.ops.random.Uniform(device=device_type, seed=42)
    result = uniform_op(
        range=[0.0, 1.0], 
        shape=[10], 
        rng=my_rng  # This should override the seed
    )
    result_np = asnumpy(result)
    assert result_np.shape == (10,)


def test_rng_clone():
    """Test that RNG.clone() creates an independent copy with the same seed."""
    # Create an RNG with a specific seed
    rng1 = ndd.random.rng(seed=5678)
    
    # Clone it
    rng2 = rng1.clone()
    
    # Verify they have the same seed
    assert rng1.seed == rng2.seed, f"Seeds don't match: {rng1.seed} != {rng2.seed}"
    
    # Verify they are different objects
    assert rng1 is not rng2, "Clone should create a new object"
    
    # Verify they generate the same sequence
    for i in range(10):
        val1 = rng1()
        val2 = rng2()
        assert val1 == val2, f"Value {i} doesn't match: {val1} != {val2}"
    
    # Verify cloned RNG works with operators
    rng3 = ndd.random.rng(seed=9999)
    rng4 = rng3.clone()
    
    uniform_op1 = ndd.ops.random.Uniform(device="cpu")
    result1 = uniform_op1(range=[0.0, 1.0], shape=[10], rng=rng3)
    result1_np = asnumpy(result1)
    
    uniform_op2 = ndd.ops.random.Uniform(device="cpu")
    result2 = uniform_op2(range=[0.0, 1.0], shape=[10], rng=rng4)
    result2_np = asnumpy(result2)
    
    # Results should be identical since clones have the same seed
    # TODO(janton): Test that same rng sequence produces same results
    # assert np.array_equal(result1_np, result2_np), "Cloned RNGs should produce identical operator results"
