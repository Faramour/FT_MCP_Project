
### [c++14_cuda](./c++14_cuda)
This sample demonstrates C++11 feature support in CUDA. It scans a input text file and prints no. of occurrences of x, y, z, w characters. 

### [concurrentKernels](./concurrentKernels)
This sample demonstrates the use of CUDA streams for concurrent execution of several kernels on GPU device. It also illustrates how to introduce dependencies between CUDA streams with the new cudaStreamWaitEvent function.


### [simpleHyperQ](./simpleHyperQ)
This sample demonstrates the use of CUDA streams for concurrent execution of several kernels on devices which provide HyperQ (SM 3.5).  Devices without HyperQ (SM 2.0 and SM 3.0) will run a maximum of two kernels concurrently.


