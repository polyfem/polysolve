set(SOURCES
	DescentStrategy.hpp
	LBFGS.hpp
	LBFGS.cpp
	BFGS.cpp
	BFGS.hpp
	GradientDescent.cpp
	GradientDescent.hpp
	Newton.hpp
	Newton.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

# ###############################################################################
# Subfolders
# ###############################################################################
# add_subdirectory(box_constraints)
