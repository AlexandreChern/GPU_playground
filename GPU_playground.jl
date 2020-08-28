using DataFrames
using CUDA
using Printf
using StaticArrays
using GPUifyLoops: @unroll

include("deriv_ops.jl")


function D2x_GPU_shared(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

    # for global memory indexing
    i = (blockIdx().x - 1) * TILE_DIM1 + tidx
    j = (blockIdx().y - 1) * TILE_DIM2 + tidy

    global_index = i + (j - 1) * Ny

    HALO_WIDTH = 1 # For second order derivative

    tile = @cuStaticSharedMem(eltype(d_u), (TILE_DIM1, TILE_DIM2 + 2 * HALO_WIDTH))

    # for tile indexing
    k = tidx
    l = tidy

    
	# Writing pencil-shaped shared memory

	# for tile itself
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx  # This statement might not be totally correct
        # fill in the code here ....

    end
    
    sync_threads()

	# for left halo
	if k <= TILE_DIM1 && l <= HALO_WIDTH && i <= Ny && HALO_WIDTH+1 <= j <= HALO_WIDTH + Nx 
        # fill in the code here ...

	end

	sync_threads()


	# for right halo

	if k <= TILE_DIM1 && TILE_DIM2 - HALO_WIDTH <= l <= TILE_DIM2 && i <= Ny && j <= Nx - HALO_WIDTH
        # fill in the code here ...
        
	end

    sync_threads()

    # Finite difference operation starts here

	# Left Boundary
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH -2 && i <= Ny && j == 1
        # fill in the code here, using data from tile (CUDA static shared memory)
        

	end

	# Center
	if k <= TILE_DIM1 && l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH - 1 && i <= Ny && 2 <= j <= Nx-1
        # fill in the code here ...
        


	end

	# Right Boundary
	if k <= TILE_DIM1 && 3 <= l + HALO_WIDTH <= TILE_DIM2 + 2*HALO_WIDTH && i <= Ny && j == Nx
        # fill in the code here ...


        
	end

    sync_threads()
    
    nothing
end


function D2y_GPU_shared(d_u, d_y, Nx, Ny, h, ::Val{TILE_DIM1}, ::Val{TILE_DIM2}) where {TILE_DIM1, TILE_DIM2}
    tidx = threadIdx().x
    tidy = threadIdx().y

	i = (blockIdx().x - 1) * TILE_DIM1 + tidx
	j = (blockIdx().y - 1) * TILE_DIM2 + tidy

	global_index = i + (j-1)*Nx

	HALO_WIDTH = 1
	tile = @cuStaticSharedMem(eltype(d_u),(TILE_DIM1+2*HALO_WIDTH,TILE_DIM2))

	k = tidx
	l = tidy

    # Writing pencil-shaped shared memory

    # for tile itself
	if k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny && j <= Nx
        # Fill in the code here ...


	end

	sync_threads()

	# For upper halo
	# if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny + HALO_WIDTH
	if k <= HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= i <= Ny && j <= Nx
       # Fill in the code here ...


	end

	sync_threads()

	# For lower halo
	# if k >= TILE_DIM1 - HALO_WIDTH && l <= TILE_DIM2 && HALO_WIDTH + 1 <= global_index <= Nx*Ny - HALO_WIDTH
	if  TILE_DIM1 - HALO_WIDTH <= k <= TILE_DIM1 && l <= TILE_DIM2 && i <= Ny - HALO_WIDTH && j <= Nx
    # Fill in the code here ...


	end

    sync_threads()
    
    # Finite Difference Operations starts 

    #Upper Boundary
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == 1 && j <= Ny
    # fill in the code here, using data from tile (CUDA static shared memory)



	end

	sync_threads()

	#Center
	if k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && 2 <= i <= Nx-1 && j <= Ny
    # fill in the code here, using data from tile (CUDA static shared memory)





	end

	sync_threads()

	#Lower Boundary
	if 3 <= k + HALO_WIDTH <= TILE_DIM1 + 2*HALO_WIDTH && l <= TILE_DIM2 && i == Nx && j <= Ny
    # fill in the code here, using data from tile (CUDA static shared memory)
          


    end
    
    sync_threads()

    nothing

end

# tester_function will test your GPU kernel against matrix-free function
# f: function name, for example D2x

function tester_function(f,Nx,TILE_DIM_1,TILE_DIM_2)
    Ny = Nx
	@show f
	# @eval gpu_function = $(Symbol(f,"_GPU"))
	@eval gpu_function_shared = $(Symbol(f,"_GPU_shared"))
	# @show gpu_function
    @show gpu_function_shared
    h = 1/Nx
	# TILE_DIM_1 = 16
	# TILE_DIM_2 = 2

	u = randn(Nx*Ny)
	d_u = CuArray(u)
	# d_y = similar(d_u)
	d_y2 = similar(d_u)

	griddim = (div(Nx,TILE_DIM_1) + 1, div(Ny,TILE_DIM_2) + 1)
	blockdim = (TILE_DIM_1,TILE_DIM_2)

	# TILE_DIM = 32
	# THREAD_NUM = TILE_DIM
    # BLOCK_NUM = div(Nx * Ny,TILE_DIM)+1 
    
	y = f(u,Nx,Ny,h)
	# @cuda threads=THREAD_NUM blocks=BLOCK_NUM gpu_function(d_u, d_y, Nx, Ny, h, Val(TILE_DIM))
    @cuda threads=blockdim blocks=griddim gpu_function_shared(d_u, d_y2, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	# @show y ≈ Array(d_y)
	@show y ≈ Array(d_y2)

	@show u
	println()
	@show y
	println()
	@show Array(d_y2)
	println()
	@show y - Array(d_y2)
	
	rep_times = 10

	t_y = time_ns()
	for i in 1:rep_times
		y = f(u,Nx,Ny,h)
	end
	t_y_end = time_ns()
	t1 = t_y_end - t_y

	memsize = length(u) * sizeof(eltype(u))
	@show Float64(t1)
	@printf("CPU Through-put %20.2f\n", 2 * memsize * rep_times / t1)


	println()

	t_d_y2 = time_ns()
	for i in 1:rep_times
		@cuda threads=blockdim blocks=griddim gpu_function_shared(d_u, d_y2, Nx, Ny, h, Val(TILE_DIM_1), Val(TILE_DIM_2))
	end
	synchronize()
	t_d_y2_end = time_ns()
	t3 = t_d_y2_end - t_d_y2

	@show Float64(t3)
	@show Float64(t1)/Float64(t3)
	@printf("GPU Through-put (shared memory)%20.2f\n", 2 * memsize * rep_times / t3)

end