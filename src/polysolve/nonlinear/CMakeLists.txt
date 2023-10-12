set(SOURCES
	LBFGS.hpp
	LBFGS.cpp
	LBFGSB.cpp
	LBFGSB.hpp
	BFGS.cpp
	BFGS.hpp
	GradientDescent.cpp
	GradientDescent.hpp
	Solver.hpp
	Solver.cpp
	Newton.hpp
	Newton.cpp
	BoxConstraintSolver.cpp
	BoxConstraintSolver.hpp
	SparseNewton.hpp
	SparseNewton.cpp
	DenseNewton.hpp
	DenseNewton.cpp
	MMA.hpp
	MMA.cpp
	MMAAux.hpp
	MMAAux.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
add_subdirectory(line_search)
