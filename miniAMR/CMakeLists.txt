cmake_minimum_required(VERSION 3.16.3)
project(miniAMR C)
find_package(MPI)
# find_package(OpenMP)

set(C_STANDARD 89)

file(GLOB sources *.c)
add_executable(ma main.c ${sources})
target_link_libraries(ma PRIVATE m MPI::MPI_C)
