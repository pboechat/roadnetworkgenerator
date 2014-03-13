file(GLOB CPP_HEADERS *.h)
file(GLOB CPP_SOURCES *.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (MSVC)
	add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
endif()

add_executable(
	statistics
	${CPP_HEADERS}
	${CPP_SOURCES}
)

set_target_properties(statistics PROPERTIES DEBUG_POSTFIX d)
