find_package(OpenGL REQUIRED)

file(GLOB CPP_HEADERS *.h)
file(GLOB CPP_SOURCES *.cpp)
file(GLOB SHADERS ${CMAKE_SOURCE_DIR}/shaders/*.glsl)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
					${OPENGL_INCLUDE}
					${CMAKE_SOURCE_DIR}/dependencies/glm/include
					${CMAKE_SOURCE_DIR}/dependencies/FreeImage/include
					${CMAKE_SOURCE_DIR}/dependencies/gl3w/include
					${CMAKE_SOURCE_DIR}/dependencies/GL_utils/include
					${CMAKE_SOURCE_DIR}/dependencies/glFont/include
					${CMAKE_SOURCE_DIR}/dependencies/vld/include
					${CMAKE_SOURCE_DIR}/src/framework/public_headers
					${CMAKE_SOURCE_DIR}/src/vector_math/public_headers
					${CMAKE_SOURCE_DIR}/src/road_network_graph/public_headers
)

source_group("Shader Files" FILES ${SHADERS})

if (MSVC)
	add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
endif (MSVC)

if (USE_QUADTREE)
	add_definitions(/D "USE_QUADTREE")
endif(USE_QUADTREE)

if (USE_GLM)
	add_definitions(/D "USE_GLM")
endif(USE_GLM)

add_executable(
	cpu_simulation
	${CPP_HEADERS}
	${CPP_SOURCES}
	gl3w.c
	${SHADERS}
)

add_custom_command(TARGET cpu_simulation POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory 
	"${PROJECT_SOURCE_DIR}/dependencies/FreeImage/lib" $<TARGET_FILE_DIR:cpu_simulation>
)

add_custom_command(TARGET cpu_simulation POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory 
	"${PROJECT_SOURCE_DIR}/dependencies/FreeImage/bin" $<TARGET_FILE_DIR:cpu_simulation>
)

target_link_libraries(
	cpu_simulation
	road_network_graph
	${OPENGL_LIBRARY}
	${OPENGL_gl_LIBRARY}
	${OPENGL_glu_LIBRARY}
	${CMAKE_SOURCE_DIR}/dependencies/FreeImage/lib/FreeImage.lib
	${CMAKE_SOURCE_DIR}/dependencies/vld/lib/vld.lib
)

set_target_properties(cpu_simulation PROPERTIES DEBUG_POSTFIX d)