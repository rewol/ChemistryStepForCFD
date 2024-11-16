cmake_minimum_required(VERSION 3.10)
project(SimpleODESolver)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 11)

# Find Sundials
#find_package(SUNDIALS REQUIRED)
include_directories(/usr/local/include/sundials)
link_directories(/usr/local/lib)

# Find LAPACK
find_package(LAPACK REQUIRED)

# Find MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# find KLU
include_directories(/usr/include/suitesparse)

# Add the executable
add_executable(ode_solver src/main.cpp)

# Link SUNDIAL Libraries (CVODES & NVector)
#target_link_libraries(ode_solver SUNDIALS::cvodes SUNDIALS::nvecserial)
target_link_libraries(ode_solver
       		     ${MPI_CXX_LIBRARIES}	
		     /usr/local/lib/libsundials_cvodes.so 
		     /usr/local/lib/libsundials_nvecserial.so
		     /usr/local/lib/libsundials_nvecparallel.so
		     /usr/local/lib/libsundials_sunlinsolklu.so
		     /usr/lib/x86_64-linux-gnu/libklu.so
		     /usr/lib/x86_64-linux-gnu/libamd.so
		     /usr/lib/x86_64-linux-gnu/libcolamd.so
		     /usr/local/lib/libsundials_core.so
		     /usr/local/lib/libsundials_sunlinsollapackdense.so
		     LAPACK::LAPACK
		     )
