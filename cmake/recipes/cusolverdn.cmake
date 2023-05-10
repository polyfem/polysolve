# cuSolverDN solver

if(TARGET CUDA::cusolver)
    return()
endif()

message(STATUS "Third-party: creating targets 'CUDA::cusolver'")

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
endif()

if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    include(cuda)
    enable_cuda(polysolve)
elseif(NOT CMAKE_CUDA_COMPILER)
    message(WARNING "CUDA not found, cuSOLVER will not be available.")
elseif(NOT CUDAToolkit_FOUND)
    message(WARNING "cuSOLVER not found, solver will not be available.")
endif()