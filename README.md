# learn CUDA in a hard way

Mainly aims at self-made operators for some interesting transformers.
# Configurations

- NVIDIA Driver: 546.92
- CUDA: 12.3
- Python: 3.10.12
- Pytorch: 2.4.0+cu118
- CMake: 3.22.1
- Ninja: 1.10.1
- GCC: 11.4.0
# Structure

```bash
.
├── LICENSE
├── README.md
├── include
│   └── add2.h # cuda kernel header of add2
├── kernel
│   └── add2_kernel.cu # cuda kernal of add2
├── lib # library so generated
├── pytorch
│   ├── CMakeLists.txt 
│   ├── add2_ops.cpp # torch wrapper for cuda kernel
│   ├── setup.py  
│   ├── time.py # time performance between self-made and torch-in
│   └── train.py # torch trainer for cuda kernel
└── tensorflow
    ├── CMakeLists.txt
    ├── add2_ops.cpp # tensorflow wrapper for cuda kernel
    ├── time.py # time performance between self-made and tensorflow-in
    └── train.py # tensorflow trainer for cuda kernel
```

# workflow

Here shows pytorch workflow with self-defined operators in CUDA, which mainly works as:
1. low-operators definition in CUDA
```cpp
// add2.h
void launch_add2(float* c, 
                const float* a,
                const float* b,
                int n);

// add2_kernel.cu
/* execution in GPU device in asychronous*/
__global__ void add2_kernel(float* c,
                            const float* a, 
                            const float* b,
                            int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] * b[i];
    }
}

void launch_add2(float* c,
                const float* a,
                const float* b,
                int n)
{
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    add2_kernel<<<grid, block>>>(c, a, b, n);
}
```
2. middle-wrappers in torch for CUDA kernel
```cpp
#include <torch/extension.h>
#include "add2.h"

/* torch wrapper for torch_launch in cpp
 * need to transform torch::Tensor to array pointer in cpp
*/
void torch_launch_add2(torch::Tensor &c,
                    const torch::Tensor &a,
                    const torch::Tensor &b,
                    int n)
{
    launch_add2((float*)c.data_ptr(),
                (const float*)a.data_ptr(),
                (const float*)b.data_ptr(),
                n);
}

/* pybind11 to bind CUDA kernel and torch wrapper for python call so*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("torch_launch_add2",
        &torch_launch_add2,
        "add2_kernel_wrapper");
}
```
3. top-python use in torch api
```python
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load


cuda_module = load(name="add2",
                sources=["add2.cpp", "add2_kernel.cu"],
                verbose=True)

n = 1024 * 1024
a = torch.rand(n, device="cuda:0")
b = torch.rand(n, device="cuda:0")
cuda_c = torch.rand(n, device="cuda:0")

ntest = 10

def show_time(func):
    times = list()
    res = list()

    for _ in range(10):
        func()

    for _ in range(ntest):
        # GPU call function in asychronous and immediately return to CPU host
        # therefore, GPU sychronization must be done at 
        # the beginning and end of timer with all kernel threads done
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        r = func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()

        times.append((end_time - start_time) * 1e6)
        res.append(r)
    
    return times, res

def run_cuda():
    cuda_module.torch_launch_add2(cuda_c, a, b, n)
    return cuda_c

def run_torch():
    a + b
    return None

cuda_time, _ = show_time(run_cuda)
print("a+b add in cuda cost {:.3f}us".format(np.mean(cuda_time)))

torch_time, _ = show_time(run_torch)
print("a+b add in torch cost {:.3f}us".format(np.mean(torch_time)))

```
## JIT
You can run the python main code directly with just-in-time.
## setuptools
```bash
python3 setup.py install
```
## CMake
```bash
mkdir build
cd build
cmake ..
make 
```

# Pytorch


# Tensorflow

