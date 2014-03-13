project(sqlite CXX C)

file(GLOB C_HEADERS include/*.h)
file(GLOB C_SOURCES src/*.c)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
  
add_library(sqlite STATIC 
	${C_HEADERS}
	${C_SOURCES}
)