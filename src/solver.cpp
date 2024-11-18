#include "solver.h"

Solver::Solver(std::shared_ptr<ProblemBase> problem, bool use_sparse, bool verbose)
    : problem_(std::move(problem)), use_sparse_(use_sparse), verbose_(verbose),
      cvode_mem_(nullptr), y_(nullptr), Jac_(nullptr), LS_(nullptr) {
    // Step 1: Create SUNDIALS context
    if (SUNContext_Create(SUN_COMM_NULL, &sunctx_) != 0) {
        throw std::runtime_error("Failed to create SUNDIALS context.");
    }

    // Step 2: Initialize the state vector
    y_ = N_VNew_Serial(problem_->getNumEquations(), sunctx_);
    if (!y_) {
        throw std::runtime_error("Failed to create state vector.");
    }

    // Set initial conditions
    const auto initial_conditions = problem_->getInitialConditions();
    for(int i = 0; i < problem_->getNumEquations(); ++i){
	    NV_Ith_S(y_, i) = initial_conditions[i];
    }
    std::cout << "No Problem up to this point " << std::endl;
    // Step 3: Set up the solver (RHS, tolerances, etc.)
    setupSolver();
}

void Solver::setupSolver() {
    // Step 1: Create the CVODE memory
    cvode_mem_ = CVodeCreate(CV_BDF, sunctx_); // Using BDF for stiff problems
    if (!cvode_mem_) {
        throw std::runtime_error("Failed to create CVODE memory.");
    }

    // Step 2: Attach the RHS function
    int flag = CVodeInit(cvode_mem_, ProblemBase::static_rhs, 0.0, y_);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeInit failed.");
    }

    // Step 3: Set tolerances
    flag = CVodeSStolerances(cvode_mem_, rel_tol_, abs_tol_);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeSStolerances failed.");
    }

    // Step 4: Set user data (pass the problem instance to RHS and Jacobian functions)
    flag = CVodeSetUserData(cvode_mem_, problem_.get());
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeSetUserData failed.");
    }

    // Step 5: Set up the Jacobian matrix and linear solver
    setupJacobian();
}

void Solver::setTolerances(sunrealtype rel_tol, sunrealtype abs_tol) {
	rel_tol_ = rel_tol;
	abs_tol_ = abs_tol;
}


void Solver::setupJacobian() {
    int num_equations = problem_->getNumEquations();

    if (use_sparse_) {
        // Step 1: Create sparse Jacobian matrix
        int num_nonzeros = problem_->getNumNonZeros();
        Jac_ = SUNSparseMatrix(num_equations, num_equations, num_nonzeros, CSR_MAT, sunctx_);
        if (!Jac_) {
            throw std::runtime_error("Failed to create sparse Jacobian matrix.");
        }

        // Step 2: Create KLU linear solver
        LS_ = SUNLinSol_KLU(y_, Jac_, sunctx_);
        if (!LS_) {
            throw std::runtime_error("Failed to create KLU linear solver.");
        }
    } else {
        // Step 1: Create dense Jacobian matrix
        Jac_ = SUNDenseMatrix(num_equations, num_equations, sunctx_);
        if (!Jac_) {
            throw std::runtime_error("Failed to create dense Jacobian matrix.");
        }

        // Step 2: Create LAPACK linear solver
        LS_ = SUNLinSol_Dense(y_, Jac_, sunctx_);
        if (!LS_) {
            throw std::runtime_error("Failed to create LAPACK linear solver.");
        }
    }

    // Step 3: Attach the linear solver to CVODE
    int flag = CVodeSetLinearSolver(cvode_mem_, LS_, Jac_);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeSetLinearSolver failed.");
    }

    // Step 4: Attach the Jacobian function
    flag = CVodeSetJacFn(cvode_mem_, ProblemBase::static_jac);
    if (flag != CV_SUCCESS) {
        throw std::runtime_error("CVodeSetJacFn failed.");
    }
}

void Solver::setTimeParameters(sunrealtype t_start, sunrealtype t_final, sunrealtype dt) {
    t_start_ = t_start;
    t_final_ = t_final;
    dt_ = dt;
}

void Solver::solve() {
    sunrealtype t = t_start_;

    while (t < t_final_) {
        int flag = CVode(cvode_mem_, t + dt_, y_, &t, CV_NORMAL);
        if (flag < 0) {
            throw std::runtime_error("CVode failed during integration.");
        }

        // Optional: Print solution
        if (verbose_) {
            std::cout << "t = " << t << ", y = [";
            for (int i = 0; i < problem_->getNumEquations(); ++i) {
                std::cout << NV_Ith_S(y_, i) << (i < problem_->getNumEquations() - 1 ? ", " : "]\n");
            }
        }
    }
}













