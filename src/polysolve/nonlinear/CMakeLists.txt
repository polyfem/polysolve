set(SOURCES
	LBFGS.hpp
	LBFGS.tpp
	LBFGSB.hpp
	BFGS.hpp
	GradientDescent.hpp
	Solver.hpp
	Solver.cpp
	BoxConstraints.hpp
	SparseNewton.hpp
	SparseNewton.tpp
	DenseNewton.hpp
	DenseNewton.tpp
	MMA.hpp
	MMA.tpp
	MMAAux.hpp
	MMAAux.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
add_subdirectory(line_search)
