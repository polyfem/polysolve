set(SOURCES
	LBFGS.hpp
	LBFGS.tpp
	LBFGSB.hpp
	BFGS.hpp
	GradientDescent.hpp
	Solver.hpp
	Solver.cpp
	Newton.hpp
	Newton.cpp
	BoxConstraints.hpp
	SparseNewton.hpp
	SparseNewton.cpp
	DenseNewton.hpp
	DenseNewton.cpp
	MMA.hpp
	MMA.tpp
	MMAAux.hpp
	MMAAux.cpp
	Utils.hpp
	Utils.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
add_subdirectory(line_search)
