set(SOURCES
	BoxConstraintSolver.cpp
	BoxConstraintSolver.hpp
	Criteria.cpp
	Criteria.hpp
	PostStepData.cpp
	PostStepData.hpp
	Problem.cpp
	Problem.hpp
	Solver.cpp
	Solver.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
add_subdirectory(descent_strategies)
add_subdirectory(line_search)
