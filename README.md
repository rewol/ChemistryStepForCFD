This code solves Hires Problem (Benchmark for stiff ODE) using CVODE of SUNDIALS.
It can work with LAPACK and KLU.

For your own ODE integration problems, you need to derive a custom class from ProblemBase class.

To Execute:
1. Create folder named build under main. Then cd to build.
2. Run "cmake .."
3. Run make
4. Execute ./ode_solver
5. Enjoy :)

Remember to check your CMakeLists.txt file.
