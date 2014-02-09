enable_testing()

file(GLOB CPP_SOURCES *.cpp)

include_directories(${CMAKE_SOURCE_DIR}/dependencies/glm/include
					${CMAKE_SOURCE_DIR}/dependencies/FreeImage/include
					${CMAKE_SOURCE_DIR}/dependencies/gl3w/include
					${CMAKE_SOURCE_DIR}/dependencies/GL_utils/include
					${gtest_SOURCE_DIR}/include 
					${gtest_SOURCE_DIR}
					${CMAKE_SOURCE_DIR}/src
					${CMAKE_CURRENT_SOURCE_DIR}
)

add_definitions(/D "USE_GLM")

add_executable(unit_tests ${CPP_SOURCES})

add_custom_command(TARGET unit_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy 
	$<TARGET_FILE:gtest> $<TARGET_FILE_DIR:unit_tests>
)

add_custom_command(TARGET unit_tests POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy 
	$<TARGET_FILE:gtest_main> $<TARGET_FILE_DIR:unit_tests>
)

target_link_libraries(
	unit_tests 
	gtest 
	gtest_main
)
