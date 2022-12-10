# cuSolverDN solver

if(TARGET CUDA::cusolver)
    return()
endif()

message(STATUS "Third-party: creating targets 'CUDA::cusolver'")

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # We do not have a build recipe for this, so find it as a system installed library.
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        set(CUDA_SEPARABLE_COMPILATION ON)
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --expt-relaxed-constexpr  -gencode arch=compute_86,code=sm_86")
        set_target_properties(polysolve
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )
        #set_property(TARGET polysolve PROPERTY CUDA_ARCHITECTURES 70 75 86)
        target_compile_options(polysolve PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
          
        --ptxas-options=-v
        --maxrregcount=7
        --compiler-options
        -fPIC # https://stackoverflow.com/questions/5311515/gcc-fpic-option
            
        >

        )
    else()
        message(WARNING "cuSOLVER not found, solver will not be available.")
    endif()
else()
    message(WARNING "CUDA not found, cuSOLVER will not be available.")
endif()