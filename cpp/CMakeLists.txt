cmake_minimum_required(VERSION 3.20)
project(FuchsUndHenneCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -Wall -Wextra -Wpedantic)
endif()

# Header-only library interface
add_library(fuchs_core INTERFACE)
target_include_directories(fuchs_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# C++ Bitboard and move engine unit tests
add_executable(test_bitboard test_bitboard.cpp)
target_link_libraries(test_bitboard PRIVATE fuchs_core)
