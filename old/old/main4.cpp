#include <iostream>
#include "sundials/sundials_types.h"	    // Basic SUNDIALS types
#include "sundials/sundials_core.h"	    // Provides core SUNDIAL types
#include "cvodes/cvodes.h"                  // Main CVODES header
#include "nvector/nvector_serial.h"         // NVector serial header
#include "nvector/nvector_parallel.h"	    // NVector parallel header
#include <sunmatrix/sunmatrix_dense.h>      // Dense matrix header 
#include <sunlinsol/sunlinsol_dense.h>      // Linear solver header (for completeness)
#include <sunmatrix/sunmatrix_sparse.h>     // Sparse matrix header
#include <sundials/sundials_context.h>      // SUNContext header for memory management
#include <sunlinsol/sunlinsol_lapackdense.h>
#include <sunlinsol/sunlinsol_klu.h>	    // KLU linear solver header
#include <chrono>			    // For measuring performance

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



int jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix Jac,
		        void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {

// Access the components of the state vector
    
     sunrealtype y1 = NV_Ith_S(y, 0);
     sunrealtype y2 = NV_Ith_S(y, 1);
     sunrealtype y3 = NV_Ith_S(y, 2);

     // Get the sparse matrix components
     sunindextype* rowptrs = SUNSparseMatrix_IndexPointers(Jac);
     sunindextype* colinds = SUNSparseMatrix_IndexValues(Jac);
     sunrealtype* data = SUNSparseMatrix_Data(Jac);

    // Set row pointers
    rowptrs[0] = 0;  // Row 0 starts at index 0
    rowptrs[1] = 3;  // Row 1 starts at index 3
    rowptrs[2] = 6;  // Row 2 starts at index 6
    rowptrs[3] = 8;  // End of row 3 (total non-zero elements)
  
    // Set column indices
    colinds[0] = 0;  // d(f1)/d(y1)
    colinds[1] = 1;  // d(f1)/d(y2)
    colinds[2] = 2;  // d(f1)/d(y3)
    colinds[3] = 0;  // d(f2)/d(y1)
    colinds[4] = 1;  // d(f2)/d(y2)
    colinds[5] = 2;  // d(f2)/d(y3)
    colinds[6] = 1;  // d(f3)/d(y2)
    colinds[7] = 2;  // d(f3)/d(y3)
    

   // Set Jacobian values (partial derivatives)
   data[0] = -0.04;                           // d(f1)/d(y1)
   data[1] = 1.0e4 * y3;                      // d(f1)/d(y2)
   data[2] = 1.0e4 * y2;                      // d(f1)/d(y3)
   data[3] = 0.04;                            // d(f2)/d(y1)
   data[4] = -1.0e4 * y3 - 6.0e7 * y2;        // d(f2)/d(y2)
   data[5] = -1.0e4 * y2;                     // d(f2)/d(y3)
   data[6] = 6.0e7 * y2;                      // d(f3)/d(y2)
   data[7] = 0.0;                             // d(f3)/d(y3)
					      //
   return 0;

}



// Main Function
int main() {
	
	// Start record
	auto start = std::chrono::high_resolution_clock::now();
	
	//MPI_Init(NULL, NULL);

	// MPI Vars
	//int num_procs, rank;
	//MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	//MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Create SUNDIALS context
	SUNContext sunctx;
	SUNContext_Create(SUN_COMM_NULL, &sunctx);
	
	// sunctx handles memory management	
	std::cout << "Flag 1 Executed" << std::endl;

	// Local & Global Vector Sizes
	//sunindextype global_length = 3;
	//sunindextype local_length = global_length / num_procs;

	// Initialize the state vector y
	//N_Vector y = N_VNew_Parallel(MPI_COMM_WORLD, local_length, global_length, sunctx);
	N_Vector y = N_VNew_Serial(3, sunctx);
        if (y == nullptr) {
            std::cerr << "Error: Failed to create N_Vector." << std::endl;
            return -1;
        }
	
	// Initial Conditions
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
	//SUNMatrix Jac = SUNDenseMatrix(N_VGetLength(y), N_VGetLength(y), sunctx);
	sunindextype NNZ = 10;	// Max number of nonzeros to be stored in a matrix
	SUNMatrix Jac = SUNSparseMatrix(N_VGetLength(y), N_VGetLength(y), NNZ, CSR_MAT, sunctx);
	std::cout << "Flag 6 Executed " << std::endl;
	

	// Create Linear Solver
	//SUNLinearSolver LS = SUNLinSol_LapackDense(y, Jac, sunctx);
	SUNLinearSolver LS = SUNLinSol_KLU(y, Jac, sunctx);  // KLU sparse solver
	
	if (LS == nullptr ) {
		std::cerr << "Error: KLU solver failed." << std::endl;
		N_VDestroy(y);
		CVodeFree(&cvode_mem);
		MPI_Finalize();
		return -1;
	}      


	if (Jac == nullptr) {
	    	std::cerr << "Error: Failed to create sparse matrix (SUNSparseMatrix)." << std::endl;
		N_VDestroy(y);
		CVodeFree(&cvode_mem);
	        MPI_Finalize();
		return -1;
	}

	int flag3 = CVodeSetLinearSolver(cvode_mem, LS, Jac);
	
	std::cout << "Flag3 is : " << flag3 << std::endl;
	std::cout << "Flag 7 Executed " << std::endl;
	
	// Atttach custom Jacobian Function
	int flagX = CVodeSetJacFn(cvode_mem, jac);
	std::cout << "FlagX is : " << flagX << std::endl;

	// Setup times for integration
	sunrealtype t 			= 0.0;
	sunrealtype t_final 		= 100;
	sunrealtype dt 			= 0.001;
		
	// Now we start the integration:
	while(t < t_final) {
		int flag = CVode(cvode_mem, t + dt, y, &t, CV_NORMAL);	// Advance the solution
		if(flag != CV_SUCCESS){
			std::cerr << "CVode Failed at t = " << t << std::endl;
			break;
		}
		
		
		std::cout << "Flag No is: " << flag << std::endl;
		std::cout << "Time t = " << t << ", [y1] = " << NV_Ith_S(y, 0)
					      << ", [y2] = " << NV_Ith_S(y, 1)
					      << ", [y3] = " << NV_Ith_S(y, 2) << std::endl;
		

	}
	
	std::cout << "Flag 8 Executed" << std::endl;

	// Free Memory
	N_VDestroy(y);			// Free state vector
	CVodeFree(&cvode_mem);		// Free the CVODE memory
	SUNContext_Free(&sunctx);	//Free the SUNDIALS context
	
	// De-allocate Linear Solver & Jacobian
	SUNLinSolFree(LS);
	SUNMatDestroy(Jac);

	std::cout << "Flag 9 Executed" << std::endl;

	//MPI_Finalize();

	// Stop Record
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> duration = end - start;

	// Output Execution Time
	std::cout << "Total Execution Time: " << duration.count() << " ms" << std::endl;

return 0;
}






























