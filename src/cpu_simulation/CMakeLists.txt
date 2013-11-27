find_package(OpenGL REQUIRED)

file(GLOB CPP_HEADERS *.h)
file(GLOB CPP_SOURCES *.cpp *.c)
file(GLOB SHADERS ${CMAKE_SOURCE_DIR}/shaders/*.glsl)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
					${OPENGL_INCLUDE}
					${CMAKE_SOURCE_DIR}/dependencies/include
					${CMAKE_SOURCE_DIR}/src/framework/public_headers
					${CMAKE_SOURCE_DIR}/src/roadnetworkgraph/public_headers
)

source_group("Shader Files" FILES ${SHADERS})

if (MSVC)
	add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
endif (MSVC)

if (USE_QUADTREE)
	add_definitions(/D "USE_QUADTREE")
endif(USE_QUADTREE)

cuda_add_executable(
	cpu_simulation
	${CPP_HEADERS}
	${CPP_SOURCES}
	${SHADERS}
)

add_custom_command(TARGET cpu_simulation POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory 
	"${PROJECT_SOURCE_DIR}/dependencies/libs" $<TARGET_FILE_DIR:cpu_simulation>
)

target_link_libraries(
	cpu_simulation
	roadnetworkgraph
	${OPENGL_LIBRARY}
	${OPENGL_gl_LIBRARY}
	${OPENGL_glu_LIBRARY}
	${CMAKE_SOURCE_DIR}/dependencies/libs/FreeImage.lib
)

set_target_properties(cpu_simulation PROPERTIES DEBUG_POSTFIX d)