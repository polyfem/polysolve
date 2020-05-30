################################################################################

if(TARGET RBF)
	return()
endif()

find_package(OpenCL)
if(NOT ${OPENCL_FOUND})
	message(WARNING "OpenCL not found; RBF interpolation will not be compiled")
	return()
endif()

################################################################################

