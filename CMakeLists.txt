cmake_minimum_required(VERSION 3.10)
project(SimpleODESolver)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 11)

# Find Sundials
#find_package(SUNDIALS REQUIRED)
include_directories(/usr/local/include/sundials)
link_directories(/usr/local/lib)

# Include directories for additional header files
include_directories(${CMAKE_SOURCE_DIR}/include)  # Include for problem.h and solver.h


# Add Python 3.10 include directory
include_directories(/usr/include/python3.10)

# Existing configuration for matplotlibcpp and Python
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# Find LAPACK
find_package(LAPACK REQUIRED)

# Find MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# find KLU
include_directories(/usr/include/suitesparse)

# Include matplotlib-cpp
include_directories(${CMAKE_SOURCE_DIR}/include)  # Add include folder for matplotlibcpp.h

# Find Python and Boost.Python
find_package(PythonLibs REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)


# Add source files
set(SOURCES
	src/main.cpp
	src/solver.cpp
	src/hires_problem.cpp  # HIRES problem implementation
	src/problem_base.cpp
	)

# Add the executable
add_executable(ode_solver ${SOURCES})

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
		     ${PYTHON_LIBRARIES}           # Link Python libraries
                     Boost::python                 # Link Boost.Python
		     )
