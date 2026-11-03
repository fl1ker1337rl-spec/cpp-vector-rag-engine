cmake_minimum_required(VERSION 3.15)
project(vector_engine)

set(CMAKE_CXX_STANDARD 17)

# CMake сам скачает pybind11 при первой сборке у любого пользователя
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(vector_engine
    core/bindings.cpp
    core/vector.cpp
)