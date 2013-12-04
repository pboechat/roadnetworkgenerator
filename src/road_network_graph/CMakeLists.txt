file(GLOB CPP_HEADERS *.h public_headers/*.h)
file(GLOB CPP_SOURCES *.cpp *.c)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
					${CMAKE_SOURCE_DIR}/dependencies/glm/include
					${CMAKE_CURRENT_SOURCE_DIR}/public_headers
					${CMAKE_SOURCE_DIR}/src/framework/public_headers
)

if (MSVC)
	add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
endif (MSVC)

if (USE_QUADTREE)
	add_definitions(/D "USE_QUADTREE")
endif(USE_QUADTREE)

add_library(
	road_network_graph STATIC
	${CPP_HEADERS}
	${CPP_SOURCES}
)

set_target_properties(road_network_graph PROPERTIES DEBUG_POSTFIX d)