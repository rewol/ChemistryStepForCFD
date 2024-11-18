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

# Add the executable
add_executable(ode_solver src/main.cpp)

# Link SUNDIAL Libraries (CVODES & NVector)
#target_link_libraries(ode_solver SUNDIALS::cvodes SUNDIALS::nvecserial)
target_link_libraries(ode_solver 
		     /usr/local/lib/libsundials_cvodes.so 
		     /usr/local/lib/libsundials_nvecserial.so
		     /usr/local/lib/libsundials_core.so
		     /usr/local/lib/libsundials_sunlinsollapackdense.so
		     LAPACK::LAPACK
		     )
