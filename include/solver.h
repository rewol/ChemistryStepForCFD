#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <memory>
#include "problem_base.h"
#include <cvodes/cvodes.h>                 // CVODE main solver
#include <sunmatrix/sunmatrix_dense.h>     // Dense matrices for LAPACK
#include <sunmatrix/sunmatrix_sparse.h>    // Sparse matrices for KLU
#include <sunlinsol/sunlinsol_dense.h>     // Linear solver for LAPACK
#include <sunlinsol/sunlinsol_klu.h>       // Sparse solver for KLU
#include <nvector/nvector_serial.h>        // Serial vectors
#include <sundials/sundials_context.h>     // SUNContext
#include <sunlinsol/sunlinsol_lapackdense.h>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

class Solver {
public:
    // Constructor
    Solver(std::shared_ptr<ProblemBase> problem, bool use_sparse, bool verbose = false);

    // Solve the problem over a specified time interval
    void solve();

    // This function sets solver tolerances based on user input
    void setTolerances(sunrealtype reltol, sunrealtype abstol);

    // This functions sets time integration parameters
    void setTimeParameters(sunrealtype tstart, sunrealtype tfinal, sunrealtype dt);

    ~Solver() = default;  // Inline definition



private:
    // Internal methods
    void setupSolver();
    void setupJacobian();

    // Member variables
    std::shared_ptr<ProblemBase> problem_; // Problem to solve
    bool use_sparse_;                      // Flag for using sparse solver (KLU) or dense solver (LAPACK)
    bool verbose_;                         // Verbosity flag for logging

    void* cvode_mem_;                      // CVODE memory
    SUNContext sunctx_;                    // SUNDIALS context
    N_Vector y_;                           // State vector
    SUNMatrix Jac_;                        // Jacobian matrix
    SUNLinearSolver LS_;                   // Linear solver
					   //
					   //
    sunrealtype rel_tol_ = 1e-4;
    sunrealtype abs_tol_ = 1e-8;

    // Time Integration
    sunrealtype t_start_ = 0.0;
    sunrealtype t_final_ = 1.0;
    sunrealtype dt_ 	= 0.01;

};

#endif // SOLVER_H

