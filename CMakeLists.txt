cmake_minimum_required(VERSION 3.15)
project(crowcpp)

set(INCLUDE_PATHS boost_1_84_0 crow/include)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ./src)

# Set up C++ wrapper for PostgreSQL C library libpq
set(SKIP_BUILD_TEST ON)


# Add compiler flags and libraries to the project
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENCV_FLAGS} -lpqxx -lpq")
set(PQXX /usr/local/include/pqxx)
find_library(PQXX_LIB pqxx)
find_library(PQ_LIB pq)

# Use file(GLOB_RECURSE ...) to add all .cpp and .h files in the src directory
file(GLOB_RECURSE SOURCES "./src/*.h")

add_executable(crowcpp main.cpp ${SOURCES})

target_compile_options(crowcpp PUBLIC "-Iinclude")
target_include_directories(crowcpp PUBLIC ${INCLUDE_PATHS})
target_link_libraries(crowcpp ${OpenCV_LIBS} ${PQXX_LIB} ${PQ_LIB})
