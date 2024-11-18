#ifndef PROBLEM_BASE_H
#define PROBLEM_BASE_H

#include <vector>
#include <sundials/sundials_types.h>    // SUNDIALS types
#include <nvector/nvector_serial.h>     // N_Vector
#include <sunmatrix/sunmatrix_sparse.h> // Sparse matrix

class ProblemBase {
public:
    virtual ~ProblemBase() = default;

    // Define the right-hand side function
    virtual int rhs(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) = 0;

    // Define the Jacobian function
    virtual int jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix Jac,
                    void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) = 0;

    // Provide the initial conditions
    virtual std::vector<sunrealtype> getInitialConditions() const = 0;

    // Precompute sparsity pattern if necessary
    virtual void precomputeSparsity() = 0;

    // Accessors for row pointers and column indices (for sparse Jacobians)
    virtual const std::vector<sunindextype>& getRowPtrs() const = 0;
    virtual const std::vector<sunindextype>& getColIndices() const = 0;

    // Returns number of equations
    virtual int getNumEquations() const = 0;
	
    virtual int getNumNonZeros() const = 0;

    // Static Functions for CVODE
    static int static_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
    static int static_jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix Jac,
                          void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);


};

#endif // PROBLEM_BASE_H
