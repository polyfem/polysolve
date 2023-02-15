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

        target_compile_options(polysolve PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:

        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true

        --ptxas-options=-v
        --maxrregcount=7
        --compiler-options
        -fPIC # https://stackoverflow.com/questions/5311515/gcc-fpic-option

        >

        )

        target_link_libraries(polysolve PUBLIC CUDA::toolkit)
        set_target_properties(polysolve PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        # Nvidia RTX8000 -> compute_75
        # Nvidia V100 -> compute_70
        # Nvidia 1080/1080Ti -> compute_61
        # Nvidia 3080Ti -> compute_86
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 70 75 86)
        endif()
        set_target_properties(polysolve PROPERTIES CUDA_ARCHITECTURES "70;75;86")

        if(APPLE)
            # We need to add the path to the driver (libcuda.dylib) as an rpath,
            # so that the static cuda runtime can find it at runtime.
            set_property(TARGET polysolve
                        PROPERTY
                        BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        endif()

    else()
        message(WARNING "cuSOLVER not found, solver will not be available.")
    endif()
else()
    message(WARNING "CUDA not found, cuSOLVER will not be available.")
endif()