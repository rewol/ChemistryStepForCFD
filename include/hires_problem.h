#ifndef HIRES_PROBLEM_H
#define HIRES_PROBLEM_H

#include "problem_base.h"
#include <vector>

class HiresProblem : public ProblemBase {
public:
    HiresProblem(); // Constructor
    ~HiresProblem() override = default;

    // Implement required virtual methods
    int rhs(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) override;
    int jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix Jac,
            void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) override;
    std::vector<sunrealtype> getInitialConditions() const override;
    void precomputeSparsity() override;
    // There are needed tı calculate Jacobian sparse matrix non zero locations once.
    const std::vector<sunindextype>& getRowPtrs() const override;
    const std::vector<sunindextype>& getColIndices() const override;

    // We need these for CVODE implementation since rhs and jac need to be static !
   // static int static_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
   // static int static_jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix Jac,
//                          void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
    
    int getNumEquations() const override {
	    return 8;
    }
    
    int getNumNonZeros() const override {
	    return 64;
    }


private:
    std::vector<sunindextype> row_ptrs_;    // Row pointers for Jacobian
    std::vector<sunindextype> col_indices_; // Column indices for Jacobian
};

#endif // HIRES_PROBLEM_H

