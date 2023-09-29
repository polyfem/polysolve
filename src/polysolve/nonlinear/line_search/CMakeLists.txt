set(SOURCES
	LineSearch.hpp
	LineSearch.cpp
	Armijo.cpp
	Armijo.hpp
	Backtracking.cpp
	Backtracking.hpp
	CppOptArmijo.hpp
	MoreThuente.hpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polysolve PRIVATE ${SOURCES})
