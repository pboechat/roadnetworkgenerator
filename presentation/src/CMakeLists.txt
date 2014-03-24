file(GLOB CPP_HEADERS *.h)
file(GLOB CPP_SOURCES *.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
					${CMAKE_SOURCE_DIR}/dependencies/glm/include
					${CMAKE_SOURCE_DIR}/src)

if (MSVC)
	add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
endif()

add_definitions(/D "USE_GLM")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DUSE_GLM")

add_executable(
	config_maker
	${CPP_HEADERS}
	${CPP_SOURCES}
)

target_link_libraries(config_maker)

set_target_properties(config_maker PROPERTIES DEBUG_POSTFIX d)
