STREAM Benchmark in CUDA C++
===========

Variant of the STREAM benchmark written in CUDA C++, based on work by Massimiliano Fatica (NVIDIA).

Three models are available: 
1) default: the original GPU STREAM using HBM. 
2) -DHOST:  use managed memory pointer allocated on the host, cuda runtime may make automatic page migration.
3) -DZERO_COPY: use coherent access in unified memory architecture, directly access host memory without data movement. 

To Do: 
add output validation. 
