# cuSolverDN solver

if(TARGET CUDA::cusolver)
    return()
endif()

message(STATUS "Third-party: creating targets 'CUDA::cusolver'")

# Use cuSolver bundled with cuda-toolkit.
find_package(CUDAToolkit)
if (CUDAToolkit_FIND)
    target_link_libraries(polysolve PUBLIC CUDA::toolkit)
    target_link_libraries(polysolve_linear PUBLIC CUDA::toolkit)
else()
    message(WARNING "cuSOLVER not found, solver will not be available.")
endif()
