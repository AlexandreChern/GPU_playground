# GPU_playground
To run use the code, you need to install the following packages
	- DataFrames
	- CUDA (only available with Julia 1.4 and above)
	- Printf
	- StaticArrays
	- GPUifyLoops

Code in this repo
	- deriv_ops.jl: this file contains CPU matrix-free operators
	- deriv_ops_GPU.jl: this file contains GPU matrix-free operators 
	- GPU_playground.jl: skeleton code that contains two GPU functions to be completed. They can be found in deriv_ops_GPU.jl, but it would be better to figure it out yourself first.
