#include <iostream>
#include <memory>
#include "solver.h"
#include "hires_problem.h"

int main() {
    try {
        
	    std::cout << "Flag 0 " << std::endl;    
	// Step 1: Create the problem instance
        std::shared_ptr<ProblemBase> problem = std::make_shared<HiresProblem>();
	    std::cout << "Flag 1 " << std::endl;
        // Step 2: Create the solver instance (use_sparse = true for KLU, false for LAPACK)
        Solver solver(problem, /*use_sparse=*/true, /*verbose=*/true);
	    std::cout << "Flag 2 " << std::endl;
        // Step 3: Optionally set tolerances
        solver.setTolerances(1e-6, 1e-10);
		std::cout << "Flag 3 " << std::endl;
        // Step 4: Optionally set time parameters
        solver.setTimeParameters(0.0, 321.8122, 1.0);
		std::cout << "Flag 4 " << std::endl;
        // Step 5: Solve the problem
        solver.solve();
		std::cout << "Flag 5 " << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

