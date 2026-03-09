cmake_minimum_required(VERSION 4.0)

project(Felina VERSION 1.0 LANGUAGES CXX)

# Force C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies first
add_subdirectory(dependencies)

# Then source files and main target
add_subdirectory(src)

# Then additional resources
add_subdirectory(shaders)
add_subdirectory(assets)
add_dependencies(Felina Shaders Assets)