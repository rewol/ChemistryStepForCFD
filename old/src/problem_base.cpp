#include "problem_base.h"

// Static RHS function
int ProblemBase::static_rhs(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    auto* problem = static_cast<ProblemBase*>(user_data);
    return problem->rhs(t, y, ydot, user_data);
}

// Static Jacobian function
int ProblemBase::static_jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix Jac,
                            void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    auto* problem = static_cast<ProblemBase*>(user_data);
    return problem->jac(t, y, fy, Jac, user_data, tmp1, tmp2, tmp3);
}

