#include <iostream>
#include "sundials/sundials_types.h"	    // Basic SUNDIALS types
#include "sundials/sundials_core.h"	    // Provides core SUNDIAL types
#include "cvodes/cvodes.h"                  // Main CVODES header
#include "nvector/nvector_serial.h"         // NVector serial header
#include <sunmatrix/sunmatrix_dense.h>      // Dense matrix header 
#include <sunlinsol/sunlinsol_dense.h>      // Linear solver header (for completeness)
#include <sundials/sundials_context.h>      // SUNContext header for memory management



// Robertson Problem

// Define rhs function
int rhs(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data) {
    sunrealtype y1 = NV_Ith_S(y, 0);
    sunrealtype y2 = NV_Ith_S(y, 1);
    sunrealtype y3 = NV_Ith_S(y, 2);

    NV_Ith_S(ydot, 0) = -0.04 * y1 + 1.0e4 * y2 * y3;
    NV_Ith_S(ydot, 1) = 0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2;
    NV_Ith_S(ydot, 2) = 3.0e7 * y2 * y2;

    return 0; // Return 0 to indicate success
}

// Main Function
int main() {
	
	int retval;
	
	// Starting main program

	// Create SUNDIALS context
	SUNContext sunctx;
	SUNContext_Create(SUN_COMM_NULL, &sunctx);
	
	// sunctx handles memory management	
	std::cout << "Flag 1 Executed" << std::endl;

	// Initialize the state vector y
	N_Vector y = N_VNew_Serial(5, sunctx);	// Creates a 1-D element vector
	
	NV_Ith_S(y, 0) = 1.0;		
	NV_Ith_S(y, 1) = 0.0;		
	NV_Ith_S(y, 2) = 0.0;	

	std::cout << "Flag 2 Executed" << std::endl;

	// Create the CVODE Solver Memory
	void* cvode_mem = CVodeCreate(CV_BDF, sunctx); // Use the BDF method for stiff problem
	if(cvode_mem == nullptr) {
		std::cerr << "CVodeCreate Failed" << std::endl;
		N_VDestroy(y);
		SUNContext_Free(&sunctx); // Free previously allocated memory
		return -1;
	}
	
	
	std::cout << "Flag 3 Executed" << std::endl;

	// Set initial time and condition
	sunrealtype t0 = 0.0;
	int flag1 = CVodeInit(cvode_mem, rhs, t0, y);

	if(flag1 != CV_SUCCESS) {
		std::cerr << "CVodeInit failed " << std::endl;
		CVodeFree(&cvode_mem);
		N_VDestroy(y);
		SUNContext_Free(&sunctx);
		return -1;
	}


	std::cout << "Flag is: " << flag1 << std::endl;
	std::cout << "Flag 4 Executed" << std::endl;

	// Set solver tolerances
	sunrealtype reltol = 1e-4;
	sunrealtype abstol = 1e-8;
	int flag2 = CVodeSStolerances(cvode_mem, reltol, abstol);


	if(flag2 != CV_SUCCESS) {
		std::cerr << "CVodeInit failed " << std::endl;
                CVodeFree(&cvode_mem);
                N_VDestroy(y);
                SUNContext_Free(&sunctx);
                return -1; 
	}

	std::cout << "Flag2 is: " << flag2 << std::endl;
	std::cout << "Flag 5 Executed" << std::endl;
	
	// Create Jacobian Matrix
	SUNMatrix Jac = SUNDenseMatrix(N_VGetLength(y), N_VGetLength(y), sunctx);
	std::cout << "Flag 6 Executed " << std::endl;
	

	// Create Linear Solver
	SUNLinearSolver LS = SUNLinSol_Dense(y, Jac, sunctx);
	int flag3 = CVodeSetLinearSolver(cvode_mem, LS, Jac);

	std::cout << "Flag3 is : " << flag3 << std::endl;
	std::cout << "Flag 7 Executed " << std::endl;
	
	// Setup times for integration
	sunrealtype t 			= 0.0;
	sunrealtype t_final 		= 100;
	sunrealtype dt 			= 0.001;
		
	// Now we start the integration:
	while(t < t_final) {
		int flag = CVode(cvode_mem, t + dt, y, &t, CV_NORMAL);	// Advance the solution
		std::cout << "Flag No is: " << flag << std::endl;
		//std::cout << "At time t = " << t << ", y = " << NV_Ith_S(y,0) << std::endl;
		
		std::cout << "Time t = " << t << ", [y1] = " << NV_Ith_S(y, 0)
					      << ", [y2] = " << NV_Ith_S(y, 1)
					      << ", [y3] = " << NV_Ith_S(y, 2) << std::endl;


	}
	
	std::cout << "Flag 8 Executed" << std::endl;

	// Free Memory
	N_VDestroy_Serial(y);	// Free state vector
	CVodeFree(&cvode_mem);	// Free the CVODE memory
	SUNContext_Free(&sunctx);	//Free the SUNDIALS context
	
	// De-allocate Linear Solver & Jacobian
	SUNLinSolFree(LS);
	SUNMatDestroy(Jac);

	std::cout << "Flag 9 Executed" << std::endl;

return 0;
}






























