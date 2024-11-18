#include <iostream>
#include <vector>
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

// Define UserData Structure
struct UserData {
	sunrealtype* dense_J;
	sunrealtype* csr_data;
	sunindextype* csr_colinds;
	sunindextype* csr_rowptrs;
	int num_species;
};

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

// A function that calculates Jacobian
void calculateJacobian(const N_Vector y, sunrealtype* dense_J) {
// Extract the state vector values
sunrealtype y1 = NV_Ith_S(y, 0);
sunrealtype y2 = NV_Ith_S(y, 1);
sunrealtype y3 = NV_Ith_S(y, 2);

// Fill the dense Jacobian matrix (row-major order)
dense_J[0] = -0.04;           dense_J[1] = 1.0e4 * y3;         dense_J[2] = 1.0e4 * y2;
dense_J[3] = 0.04;            dense_J[4] = -1.0e4 * y3 - 6.0e7 * y2; dense_J[5] = -1.0e4 * y2;
dense_J[6] = 0.0;             dense_J[7] = 6.0e7 * y2;         dense_J[8] = 0.0;
}

// User Defined Jacobian Function
int jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix Jac,
		        void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
     // Cast User Data 
     UserData* data = static_cast<UserData*>(user_data);
     // Calculate the dense Jacobian using the current state vector
     calculateJacobian(y, data->dense_J);

     // Convert the dense Jacobian to CSR format
     int nnz = 0;
     for (int i = 0; i < data->num_species; ++i) {
 	    for (int j = 0; j < data->num_species; ++j) {
     	       sunrealtype value = data->dense_J[i * data->num_species + j];
               if (value != 0.0) {
              	 data->csr_data[nnz] = value; // Update non-zero value
                 nnz++;
               }
            }
     }
     // Update the SUNMatrix with the new CSR data
     sunrealtype* J_data = SUNSparseMatrix_Data(Jac);  // Access CSR non-zero values in J
     std::copy(data->csr_data, data->csr_data + nnz, J_data);

     return 0;

}



// Main Function
int main() {
	
	// Start record
	auto start = std::chrono::high_resolution_clock::now();
	
	const int num_species = 3;
		
	// Create SUNDIALS context
	SUNContext sunctx;
	SUNContext_Create(SUN_COMM_NULL, &sunctx);
	
	// sunctx handles memory management	
	std::cout << "Flag 1 Executed" << std::endl;
	
	// Create Vector y & Initialize
	N_Vector y = N_VNew_Serial(num_species, sunctx);
        if (y == nullptr) {
            std::cerr << "Error: Failed to create N_Vector." << std::endl;
            return -1;
        }
	
	// Initial Conditions
	NV_Ith_S(y, 0) = 1.0;		
	NV_Ith_S(y, 1) = 0.0;		
	NV_Ith_S(y, 2) = 0.0;

	std::cout << "Flag 2 Executed" << std::endl;

	// Create & Define User Data
	UserData user_data;
	user_data.num_species = num_species;
	// Allocate Jacobian Matrix
	user_data.dense_J = new sunrealtype[num_species * num_species];
	// Allocate csr arrays
	int nnz = 7;	// This is the number of nonzero elements
        user_data.csr_data = new sunrealtype[nnz];
        user_data.csr_colinds = new sunindextype[nnz];
        user_data.csr_rowptrs = new sunindextype[num_species + 1];
	// Set Row Pointers & Columns Pointers
        user_data.csr_rowptrs[0] = 0;
        user_data.csr_rowptrs[1] = 3;
        user_data.csr_rowptrs[2] = 6;
        user_data.csr_rowptrs[3] = 7;

        user_data.csr_colinds[0] = 0; user_data.csr_colinds[1] = 1; user_data.csr_colinds[2] = 2;
        user_data.csr_colinds[3] = 0; user_data.csr_colinds[4] = 1; user_data.csr_colinds[5] = 2;
        user_data.csr_colinds[6] = 1;

	// Create the CVODE Solver Memory
	void* cvode_mem = CVodeCreate(CV_BDF, sunctx); // Use the BDF method for stiff problem
	if(cvode_mem == nullptr) {
		std::cerr << "CVodeCreate Failed" << std::endl;
		N_VDestroy(y);
		SUNContext_Free(&sunctx); // Free previously allocated memory
		return -1;
	}
	
	std::cout << "Flag 3 Executed" << std::endl;

	// Set User Data
	int flagUD = CVodeSetUserData(cvode_mem, &user_data); // This is how we establish user_data in jac fnc
	std::cout << "flagUD is " << flagUD << std::endl;

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
	
	// Do some clean-up
	delete[] user_data.dense_J;
	delete[] user_data.csr_data;
	delete[] user_data.csr_colinds;
	delete[] user_data.csr_rowptrs;


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






























