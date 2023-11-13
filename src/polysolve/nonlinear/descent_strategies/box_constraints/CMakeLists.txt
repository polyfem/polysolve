set(SOURCES
	LBFGSB.cpp
	LBFGSB.hpp
	BoxedDescentStrategy.hpp
	MMA.hpp
	MMA.cpp
	MMAAux.hpp
	MMAAux.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})

