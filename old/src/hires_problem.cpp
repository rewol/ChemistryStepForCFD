#include "hires_problem.h"
#include <cmath> // For mathematical functions like exp

HiresProblem::HiresProblem() {
    // Initialize the sparsity pattern
    precomputeSparsity();
}

// Instance-specific rhs function
int HiresProblem::rhs(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    // Access state variables
    sunrealtype y1 = NV_Ith_S(y, 0);
    sunrealtype y2 = NV_Ith_S(y, 1);
    sunrealtype y3 = NV_Ith_S(y, 2);
    sunrealtype y4 = NV_Ith_S(y, 3);
    sunrealtype y5 = NV_Ith_S(y, 4);
    sunrealtype y6 = NV_Ith_S(y, 5);
    sunrealtype y7 = NV_Ith_S(y, 6);
    sunrealtype y8 = NV_Ith_S(y, 7);

    // Compute derivatives
    NV_Ith_S(ydot, 0) = -1.71 * y1 + 0.43 * y2 + 8.32 * y3 + 0.0007 * y4;
    NV_Ith_S(ydot, 1) = 1.71 * y1 - 8.75 * y2;
    NV_Ith_S(ydot, 2) = -10.03 * y3 + 0.43 * y4 + 0.035 * y5;
    NV_Ith_S(ydot, 3) = 8.32 * y2 + 1.71 * y3 - 1.12 * y4;
    NV_Ith_S(ydot, 4) = -1.745 * y5 + 0.43 * y6 + 0.43 * y7;
    NV_Ith_S(ydot, 5) = -280.0 * y6 * y8 + 0.69 * y4 + 1.71 * y5 - 0.43 * y6 + 0.69 * y7;
    NV_Ith_S(ydot, 6) = 280.0 * y6 * y8 - 1.81 * y7;
    NV_Ith_S(ydot, 7) = -280.0 * y6 * y8 + 1.81 * y7;

    return 0;
}

int HiresProblem::jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix Jac,
                      void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    // Access state variables
    sunrealtype y6 = NV_Ith_S(y, 5);  // y6 is the sixth variable
    sunrealtype y8 = NV_Ith_S(y, 7);  // y8 is the eighth variable

    // Determine the type of the Jacobian matrix
    SUNMatrix_ID matrix_id = SUNMatGetID(Jac);

    if (matrix_id == SUNMATRIX_DENSE) {
        // Dense Matrix (LAPACK)
        sunrealtype* Jdata = SUNDenseMatrix_Data(Jac);
        sunindextype cols = SUNDenseMatrix_Columns(Jac);

        // Fill the dense Jacobian row by row (all elements, including zeros)
        Jdata[0 * cols + 0] = -1.71;   // d(f1)/d(y1)
        Jdata[0 * cols + 1] = 0.43;    // d(f1)/d(y2)
        Jdata[0 * cols + 2] = 8.32;    // d(f1)/d(y3)
        Jdata[0 * cols + 3] = 0.0007;  // d(f1)/d(y4)
        Jdata[0 * cols + 4] = 0.0;     // d(f1)/d(y5)
        Jdata[0 * cols + 5] = 0.0;     // d(f1)/d(y6)
        Jdata[0 * cols + 6] = 0.0;     // d(f1)/d(y7)
        Jdata[0 * cols + 7] = 0.0;     // d(f1)/d(y8)

        Jdata[1 * cols + 0] = 1.71;    // d(f2)/d(y1)
        Jdata[1 * cols + 1] = -8.75;   // d(f2)/d(y2)
        Jdata[1 * cols + 2] = 0.0;     // d(f2)/d(y3)
        Jdata[1 * cols + 3] = 0.0;     // d(f2)/d(y4)
        Jdata[1 * cols + 4] = 0.0;     // d(f2)/d(y5)
        Jdata[1 * cols + 5] = 0.0;     // d(f2)/d(y6)
        Jdata[1 * cols + 6] = 0.0;     // d(f2)/d(y7)
        Jdata[1 * cols + 7] = 0.0;     // d(f2)/d(y8)

        Jdata[2 * cols + 0] = 0.0;     // d(f3)/d(y1)
        Jdata[2 * cols + 1] = 0.43;    // d(f3)/d(y2)
        Jdata[2 * cols + 2] = -10.03;  // d(f3)/d(y3)
        Jdata[2 * cols + 3] = 0.43;    // d(f3)/d(y4)
        Jdata[2 * cols + 4] = 0.035;   // d(f3)/d(y5)
        Jdata[2 * cols + 5] = 0.0;     // d(f3)/d(y6)
        Jdata[2 * cols + 6] = 0.0;     // d(f3)/d(y7)
        Jdata[2 * cols + 7] = 0.0;     // d(f3)/d(y8)

        Jdata[3 * cols + 0] = 0.0;     // d(f4)/d(y1)
        Jdata[3 * cols + 1] = 8.32;    // d(f4)/d(y2)
        Jdata[3 * cols + 2] = 1.71;    // d(f4)/d(y3)
        Jdata[3 * cols + 3] = -1.12;   // d(f4)/d(y4)
        Jdata[3 * cols + 4] = 0.0;     // d(f4)/d(y5)
        Jdata[3 * cols + 5] = 0.0;     // d(f4)/d(y6)
        Jdata[3 * cols + 6] = 0.0;     // d(f4)/d(y7)
        Jdata[3 * cols + 7] = 0.0;     // d(f4)/d(y8)

        Jdata[4 * cols + 0] = 0.0;     // d(f5)/d(y1)
        Jdata[4 * cols + 1] = 0.0;     // d(f5)/d(y2)
        Jdata[4 * cols + 2] = 0.0;     // d(f5)/d(y3)
        Jdata[4 * cols + 3] = 0.43;    // d(f5)/d(y4)
        Jdata[4 * cols + 4] = -1.745;  // d(f5)/d(y5)
        Jdata[4 * cols + 5] = 0.43;    // d(f5)/d(y6)
        Jdata[4 * cols + 6] = 0.43;    // d(f5)/d(y7)
        Jdata[4 * cols + 7] = 0.0;     // d(f5)/d(y8)

        Jdata[5 * cols + 0] = 0.0;     // d(f6)/d(y1)
        Jdata[5 * cols + 1] = 0.0;     // d(f6)/d(y2)
        Jdata[5 * cols + 2] = 0.0;     // d(f6)/d(y3)
        Jdata[5 * cols + 3] = 0.69;    // d(f6)/d(y4)
        Jdata[5 * cols + 4] = 1.71;    // d(f6)/d(y5)
        Jdata[5 * cols + 5] = -280.0 * y8 - 0.43; // d(f6)/d(y6)
        Jdata[5 * cols + 6] = 0.69;    // d(f6)/d(y7)
        Jdata[5 * cols + 7] = -280.0 * y6; // d(f6)/d(y8)

        Jdata[6 * cols + 0] = 0.0;     // d(f7)/d(y1)
        Jdata[6 * cols + 1] = 0.0;     // d(f7)/d(y2)
        Jdata[6 * cols + 2] = 0.0;     // d(f7)/d(y3)
        Jdata[6 * cols + 3] = 0.0;     // d(f7)/d(y4)
        Jdata[6 * cols + 4] = 0.0;     // d(f7)/d(y5)
        Jdata[6 * cols + 5] = 280.0 * y8; // d(f7)/d(y6)
        Jdata[6 * cols + 6] = -1.81;   // d(f7)/d(y7)
        Jdata[6 * cols + 7] = 280.0 * y6; // d(f7)/d(y8)

        Jdata[7 * cols + 0] = 0.0;     // d(f8)/d(y1)
        Jdata[7 * cols + 1] = 0.0;     // d(f8)/d(y2)
        Jdata[7 * cols + 2] = 0.0;     // d(f8)/d(y3)
        Jdata[7 * cols + 3] = 0.0;     // d(f8)/d(y4)
        Jdata[7 * cols + 4] = 0.0;     // d(f8)/d(y5)
        Jdata[7 * cols + 5] = -280.0 * y8; // d(f8)/d(y6)
        Jdata[7 * cols + 6] = 1.81;    // d(f8)/d(y7)
        Jdata[7 * cols + 7] = -280.0 * y6; // d(f8)/d(y8)
    } else if (matrix_id == SUNMATRIX_SPARSE) {
        // Sparse Matrix (KLU)
        // You’ll map only non-zero values based on sparsity
	
    
    sunrealtype* Jdata = SUNSparseMatrix_Data(Jac);
    sunindextype* rowptrs = SUNSparseMatrix_IndexPointers(Jac);
    sunindextype* colinds = SUNSparseMatrix_IndexValues(Jac);
    // Use precomputed sparsity pattern for rowptrs and colinds
    std::copy(row_ptrs_.begin(), row_ptrs_.end(), rowptrs);
    std::copy(col_indices_.begin(), col_indices_.end(), colinds);
    
    // Fill the non-zero entries of the sparse Jacobian matrix
    Jdata[0] = -1.71;   // d(f1)/d(y1)
    Jdata[1] = 0.43;    // d(f1)/d(y2)
    Jdata[2] = 8.32;    // d(f1)/d(y3)
    Jdata[3] = 0.0007;  // d(f1)/d(y4)

    Jdata[4] = 1.71;    // d(f2)/d(y1)
    Jdata[5] = -8.75;   // d(f2)/d(y2)

    Jdata[6] = 0.43;    // d(f3)/d(y2)
    Jdata[7] = -10.03;  // d(f3)/d(y3)
    Jdata[8] = 0.43;    // d(f3)/d(y4)
    Jdata[9] = 0.035;   // d(f3)/d(y5)

    Jdata[10] = 8.32;   // d(f4)/d(y2)
    Jdata[11] = 1.71;   // d(f4)/d(y3)
    Jdata[12] = -1.12;  // d(f4)/d(y4)

    Jdata[13] = 0.43;   // d(f5)/d(y4)
    Jdata[14] = -1.745; // d(f5)/d(y5)
    Jdata[15] = 0.43;   // d(f5)/d(y6)
    Jdata[16] = 0.43;   // d(f5)/d(y7)

    Jdata[17] = 0.69;    // d(f6)/d(y4)
    Jdata[18] = 1.71;    // d(f6)/d(y5)
    Jdata[19] = -280.0 * y8 - 0.43; // d(f6)/d(y6)
    Jdata[20] = 0.69;    // d(f6)/d(y7)
    Jdata[21] = -280.0 * y6; // d(f6)/d(y8)

    Jdata[22] = 280.0 * y8; // d(f7)/d(y6)
    Jdata[23] = -1.81;      // d(f7)/d(y7)
    Jdata[24] = 280.0 * y6; // d(f7)/d(y8)

    Jdata[25] = -280.0 * y8; // d(f8)/d(y6)
    Jdata[26] = 1.81;        // d(f8)/d(y7)
    Jdata[27] = -280.0 * y6; // d(f8)/d(y8ccess
}

return 0;
}

/*
// Static wrapper for rhs
int HiresProblem::static_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    auto* problem = static_cast<HiresProblem*>(user_data);
    return problem->rhs(t, y, ydot, user_data);
}

// Static wrapper for jac
int HiresProblem::static_jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix Jac,
                             void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    auto* problem = static_cast<HiresProblem*>(user_data);
    return problem->jac(t, y, fy, Jac, user_data, tmp1, tmp2, tmp3);
}
*/


// Get initial conditions
std::vector<sunrealtype> HiresProblem::getInitialConditions() const {
    return {1.0, 0.0, 0.0, 0.0, 0.0, 0.0057, 0.0, 0.0};
}

// Precompute sparsity pattern
void HiresProblem::precomputeSparsity() {
    row_ptrs_ = {0, 4, 6, 10, 13, 17, 22, 25, 28};
    col_indices_ = {0, 1, 2, 3, 0, 1, 1, 2, 3, 4,
                    1, 2, 3, 3, 4, 5, 6, 3, 4, 5,
                    6, 7, 5, 6, 7, 5, 6, 7};
}

// Access row pointers
const std::vector<sunindextype>& HiresProblem::getRowPtrs() const {
    return row_ptrs_;
}

// Access column indices
const std::vector<sunindextype>& HiresProblem::getColIndices() const {
    return col_indices_;
}












