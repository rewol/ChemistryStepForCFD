#include <iostream>
#include "sundials/sundials_types.h"	    // Basic SUNDIALS types
#include "sundials/sundials_core.h"	    // Provides core SUNDIAL types
#include "cvodes/cvodes.h"                  // Main CVODES header
#include "nvector/nvector_serial.h"         // NVector serial header
#include <sunmatrix/sunmatrix_dense.h>      // Dense matrix header 
#include <sunlinsol/sunlinsol_dense.h>      // Linear solver header (for completeness)
#include <sundials/sundials_context.h>      // SUNContext header for memory management

// Right-hand side function for the ODE: dy/dt = -0.5 * y

int rhs(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data) {
       
	if(y == nullptr || ydot == nullptr)
	{
		std::cerr << "Null Ptr in rhs function" << std::endl;
		return -1;
	}
	
	
       sunrealtype y_val = NV_Ith_S(y, 0);    // Access the first component of y
       NV_Ith_S(ydot, 0) = -0.5 * y_val;   // dy/dt = -0.5 * y
       
       return 0; // Return 0 to indicate success
   }


// Main Function
int main() {
	
	int retval;

	// Create SUNDIALS context
	SUNContext sunctx;
	SUNContext_Create(SUN_COMM_NULL, &sunctx);
	
	std::cout << "Flag 1 Executed" << std::endl;

	// Initialize the state vectıor y
	N_Vector y = N_VNew_Serial(1, sunctx);	// Creates a 1-D element vector
	NV_Ith_S(y, 0) = 1.0;		// Set initial condition to y(0) = 1
	
	std::cout << "Flag 2 Executed" << std::endl;

	// Create the CVODE Solver Memory
	void* cvode_mem = CVodeCreate(CV_ADAMS, sunctx); //Use the adams method for non-stiff problem
	//void* cvode_mem = NULL;
	//cvode_mem = CVodeCreate(CV_ADAMS, CV_NEWTON);
	if(cvode_mem == nullptr) {
		std::cerr << "CVodeCreate Failed" << std::endl;
		return -1;
	}
	
	
	std::cout << "Flag 3 Executed" << std::endl;

	// Set initial time and condition
	sunrealtype t0 = 0.0;
	int flag1 = CVodeInit(cvode_mem, rhs, t0, y);
	std::cout << "Flag is: " << flag1 << std::endl;
	std::cout << "Flag 4 Executed" << std::endl;

	// Set solver tolerances
	sunrealtype tol1 = 1e-4;
	sunrealtype tol2 = 1e-8;
	int flag2 = CVodeSStolerances(cvode_mem, tol1, tol2);
	std::cout << "Flag2 is: " << flag2 << std::endl;
	std::cout << "Flag 5 Executed" << std::endl;

	// Integrate the ODE up to t_final = 1.0
	sunrealtype t = 0.0;
	sunrealtype t_final = 1.0;
	
	// Create Jacobian Matrix
	SUNMatrix Jac = SUNDenseMatrix(N_VGetLength(y), N_VGetLength(y), sunctx);
	// Create Linear Solver
	SUNLinearSolver LS = SUNLinSol_Dense(y, Jac, sunctx);
	CVodeSetLinearSolver(cvode_mem, LS, Jac);

		
	// Now we start the integration:
	while(t < t_final) {
		int flag = CVode(cvode_mem, t_final, y, &t, CV_NORMAL);	// Advance the solution
		std::cout << "Flag No is: " << flag << std::endl;
		std::cout << "At time t = " << t << ", y = " << NV_Ith_S(y,0) << std::endl;
	}
	
	std::cout << "Flag 6 Executed" << std::endl;

	// Free Memory
	N_VDestroy_Serial(y);	// Free state vector
	CVodeFree(&cvode_mem);	// Free the CVODE memory
	SUNContext_Free(&sunctx);	//Free the SUNDIALS context
	
	// De-allocate Linear Solver & Jacobian
	SUNLinSolFree(LS);
	SUNMatDestroy(Jac);

	std::cout << "Flag 7 Executed" << std::endl;

return 0;
}






























