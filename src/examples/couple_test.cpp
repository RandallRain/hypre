/**
* @file couple_test.cpp
* @brief Test BoomerAMG ability to solve the System of PDEs using 2D Stokes problem.
* There are three forms of matrix:
* - The saddle point form.
* - the weakly coupled form.
* - the strongly coupled form.
* @author LHT
* @date 2025-03-19
*/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>

/* Hypre headers */
#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"

// The matrix form of 2D Stokes problem.
enum class MatrixForm 
{
    SaddlePoint = 0,
    WeaklyCoupled = 1,
    StronglyCoupled = 2
};

// Hypre BoomerAMG solver types.
enum class SolverType
{
    UnknownBased = 0,
    NodeBased = 1
};

// Global variables.
int g_pde_size = 3; ///< number of unknowns at each grid point.
MatrixForm g_matrix_form = MatrixForm::WeaklyCoupled; ///< matrix form.
SolverType g_solver_type = SolverType::NodeBased; ///< solver type.
bool g_print_matrix = false; ///< print the matrix to a file.

/**
 * @brief Create a linear system according to the 2D Stokes problem.
 * With the no-slip wall boundary conditions, and center force applied.
 * @param[in, out] A a reference to the matrix.
 * @param[in, out] b a reference to the RHS vector.
 * @param[in, out] x a reference to the solution vector.
 * @param n the number grid steps(n+1 grid points) in each direction.
 * The grid point index: [0, n], the unkonwn grid points: [1, n-1].
 * @param matrix_form the matrix form, using the MatrixForm enum.
 * @param comm MPI communicator.
 * @param my_rank my process rank.
 * @param num_procs the number of processes.
 */
void create_2d_stokes_system(HYPRE_IJMatrix& A, HYPRE_IJVector& b, HYPRE_IJVector& x, 
                             int n, MatrixForm matrix_form, MPI_Comm comm, int my_rank, int num_procs)
{
    int system_size = g_pde_size*(n-1)*(n-1); ///< total number of unknowns.
    double h = 1.0 / n; ///< grid spacing.
    double h2_inv = 1.0 / (h * h);
    
    // Determine local range for current processor, divided by rows.
    // [local_start, local_end]
    int local_size = system_size / num_procs;
    int extra = system_size % num_procs;
    int local_start, local_end;
    
    // Redistribute the extra rows to the first processors.
    if (my_rank < extra) 
    {
        local_size++;
        local_start = my_rank * local_size;
    } else 
    {
        local_start = my_rank * (local_size) + extra;
    }
    local_end = local_start + local_size - 1;
    
    std::cout << "Processor " << my_rank << " handles rows " << local_start << " to " << local_end << std::endl;
    
    // Create the matrix, RHS, and solution vector.
    HYPRE_IJMatrixCreate(comm, local_start, local_end, local_start, local_end, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJVectorCreate(comm, local_start, local_end, &b);
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorCreate(comm, local_start, local_end, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    
    // Initialize the matrix and vectors.
    HYPRE_IJMatrixInitialize(A);
    HYPRE_IJVectorInitialize(b);
    HYPRE_IJVectorInitialize(x);
    
    // Calulate the index for u, v, and p at grid point (i, j).
    auto U_IDX = [n](int i, int j) { return ((i-1)*(n-1) + (j-1)) * g_pde_size; };
    auto V_IDX = [n](int i, int j) { return ((i-1)*(n-1) + (j-1)) * g_pde_size + 1; };
    auto P_IDX = [n](int i, int j) { return ((i-1)*(n-1) + (j-1)) * g_pde_size + 2; };

    // Set up the matrix and vectors, loop over all to-be-solved grid points.
    // For strongly coupled PDEs, use node-based indexing.
    for (int i = 1; i < n; i++) 
    {
        for (int j = 1; j < n; j++) 
        {
            int idx_u = U_IDX(i, j); ///< (i, j) index for u velocity.
            int idx_v = V_IDX(i, j); ///< (i, j) index for v velocity.
            int idx_p = P_IDX(i, j); ///< (i, j) index for pressure.
            
            // Build u-velocity equation row at grid (i, j).
            if (idx_u >= local_start && idx_u <= local_end)
            {
                std::vector<int> cols; ///< column indices.
                std::vector<double> values; ///< values.
                
                // Laplacian stencil.
                // (i, j)
                cols.push_back(U_IDX(i, j));
                values.push_back(4.0 * h2_inv);
                
                // (i-1, j) interior points.
                if (i > 1)
                {
                    cols.push_back(U_IDX(i-1, j));
                    values.push_back(-1.0 * h2_inv);
                }
                
                // (i+1, j) interior points.
                if (i < n-1)
                {
                    cols.push_back(U_IDX(i+1, j));
                    values.push_back(-1.0 * h2_inv);
                }

                // (i, j-1) interior points.
                if (j > 1)
                {
                    cols.push_back(U_IDX(i, j-1));
                    values.push_back(-1.0 * h2_inv);
                }

                // (i, j+1) interior points.
                if (j < n-1)
                {
                    cols.push_back(U_IDX(i, j+1));
                    values.push_back(-1.0 * h2_inv);
                }
                
                // Pressure gradient in x-direction.
                // (i+1, j) interior points.
                if (i < n-1)
                {
                    cols.push_back(P_IDX(i+1, j));
                    if (matrix_form == MatrixForm::SaddlePoint ||
                        matrix_form == MatrixForm::WeaklyCoupled)
                    {
                        values.push_back(0.5 / h);
                    }
                    if (matrix_form == MatrixForm::StronglyCoupled)                  
                    {
                        // Enforce strong coupling.
                        values.push_back(1.0 * h2_inv);
                    }
                }
                // (i-1, j) interior points.
                if (i > 1)
                {
                    cols.push_back(P_IDX(i-1, j));
                    if (matrix_form == MatrixForm::SaddlePoint ||
                        matrix_form == MatrixForm::WeaklyCoupled)
                    {
                        values.push_back(-0.5 / h);
                    }
                    if (matrix_form == MatrixForm::StronglyCoupled)                  
                    {
                        // Enforce strong coupling.
                        values.push_back(-1.0 * h2_inv);
                    }
                }

                // Set matrix row.
                int num_cols = cols.size();
                HYPRE_IJMatrixSetValues(A, 1, &num_cols, &idx_u, cols.data(), values.data());
                
                // Set RHS and initial guess.
                // Apply a force in the center.
                double rhs_val = (i == n/2 && j == n/2) ? 1.0 : 0.0;
                double zero = 0.0;
                
                HYPRE_IJVectorSetValues(b, 1, &idx_u, &rhs_val);
                HYPRE_IJVectorSetValues(x, 1, &idx_u, &zero);
            }
            
            // Build v-velocity equation row at grid point (i, j).
            if (idx_v >= local_start && idx_v <= local_end) 
            {
                std::vector<int> cols;
                std::vector<double> values;
                
                // Laplacian stencil.
                // (i, j)
                cols.push_back(V_IDX(i, j));
                values.push_back(4.0 * h2_inv);
                
                // (i-1, j) interior points.
                if (i > 1)
                {
                    cols.push_back(V_IDX(i-1, j));
                    values.push_back(-1.0 * h2_inv);
                }
                
                // (i+1, j) interior points.
                if (i < n-1)
                {
                    cols.push_back(V_IDX(i+1, j));
                    values.push_back(-1.0 * h2_inv);
                }

                // (i, j-1) interior points.
                if (j > 1)
                {
                    cols.push_back(V_IDX(i, j-1));
                    values.push_back(-1.0 * h2_inv);
                }

                // (i, j+1) interior points.
                if (j < n-1)
                {
                    cols.push_back(V_IDX(i, j+1));
                    values.push_back(-1.0 * h2_inv);
                }
                
                // Pressure gradient in y-direction.
                // (i, j+1) interior points.
                if (j < n-1)
                {
                    cols.push_back(P_IDX(i, j+1));
                    switch (matrix_form)
                    {
                        case MatrixForm::SaddlePoint:
                        case MatrixForm::WeaklyCoupled:
                            values.push_back(0.5 / h);
                            break;
                        case MatrixForm::StronglyCoupled:
                            // Enforce strong coupling.
                            values.push_back(1.0 * h2_inv);
                            break;
                    }
                }
                // (i, j-1) interior points.
                if (j > 1)
                {
                    cols.push_back(P_IDX(i, j-1));
                    switch (matrix_form)
                    {
                        case MatrixForm::SaddlePoint:
                        case MatrixForm::WeaklyCoupled:
                            values.push_back(-0.5 / h);
                            break;
                        case MatrixForm::StronglyCoupled:
                            // Enforce strong coupling.
                            values.push_back(-1.0 * h2_inv);
                            break;
                    }
                }
                // Set matrix row.
                int num_cols = cols.size();
                HYPRE_IJMatrixSetValues(A, 1, &num_cols, &idx_v, cols.data(), values.data());

                // Set RHS and initial guess.
                double rhs_val = 0.0;
                double zero = 0.0;
                
                HYPRE_IJVectorSetValues(b, 1, &idx_v, &rhs_val);
                HYPRE_IJVectorSetValues(x, 1, &idx_v, &zero);
            }

            // Build pressure equation (continuity) row at grid (i, j).
            if (idx_p >= local_start && idx_p <= local_end) 
            {
                std::vector<int> cols; ///< column indices.
                std::vector<double> values; ///< values.
                
                // Divergence operator, \frac{\partial u}{\partial x}.
                // (i-1, j) interior points.
                if (i > 1)
                {
                    cols.push_back(U_IDX(i-1, j));
                    values.push_back(-0.5 / h);
                }
                
                // (i+1, j) interior points.
                if (i < n-1)
                {
                    cols.push_back(U_IDX(i+1, j));
                    values.push_back(0.5 / h);
                }
                
                // Divergence operator, \frac{\partial v}{\partial y}.
                // (i, j-1) interior points.
                if (j > 1)
                {
                    cols.push_back(V_IDX(i, j-1));
                    values.push_back(-0.5 / h);
                }
                
                // (i, j+1) interior points.
                if (j < n-1)
                {
                    cols.push_back(V_IDX(i, j+1));
                    values.push_back(0.5 / h);
                }
                
                // Handle Saddle point structure.
                cols.push_back(P_IDX(i, j));
                switch (matrix_form)
                {
                    case MatrixForm::SaddlePoint:
                        // Very small number, add saddle point structure.
                        values.push_back(1e-3);
                        break;
                    case MatrixForm::WeaklyCoupled:
                        // The same scale as the divergence operator, no saddle point structure.
                        values.push_back(1.0 * h2_inv);
                        break;
                    case MatrixForm::StronglyCoupled:
                        // The same scale as the divergence operator, no saddle point structure.
                        // Enfore strong coupling.
                        values.push_back(0.5 / h);
                        break;
                }
                
                // Set matrix row.
                int num_cols = cols.size();
                HYPRE_IJMatrixSetValues(A, 1, &num_cols, &idx_p, cols.data(), values.data());
                
                // Set RHS and initial guess.
                double rhs_val = 0.0;
                double zero = 0.0;
                
                HYPRE_IJVectorSetValues(b, 1, &idx_p, &rhs_val);
                HYPRE_IJVectorSetValues(x, 1, &idx_p, &zero);
            }
        }
    }
    
    // Assemble the matrix and vectors.
    HYPRE_IJMatrixAssemble(A);
    HYPRE_IJVectorAssemble(b);
    HYPRE_IJVectorAssemble(x);
}

// Main functions.
int main(int argc, char *argv[])
{
    int myid; ///< my process rank.
    int num_procs; ///< num of processors.
    int n = 32; ///< num of grid steps per dimension.

    double create_time; ///< time to create the system, seconds.
    double setup_time; ///< time to setup the solver, seconds.
    double solve_time; ///< time to solve the system, seconds.
    clock_t start, end;
    
    // Initialize MPI.
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    // Parse command line arguments.
    for (int i = 1; i < argc; i++) 
    {
        // num of grid steps per dimension.
        if (strcmp(argv[i], "-n") == 0) 
        {
            n = atoi(argv[i+1]);
            i++;
        }
    }
    
    // Initialize Hypre.
    HYPRE_Init();
    
    /* Create the matrix, RHS and solution */
    HYPRE_IJMatrix A;
    HYPRE_IJVector b, x;
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_ParVector par_b, par_x;
    
    if (myid == 0) {
        std::cout << "Testing BoomerAMG on 2D Stokes problem" << std::endl;
        std::cout << "Grid size: " << (n + 1) << " x " << (n + 1) << std::endl;
        std::cout << "System size(Unknows): " << (3*(n-1)*(n-1)) << std::endl;
    }
    
    // Create the Stokes system.
    start = clock();
    create_2d_stokes_system(A, b, x, n, g_matrix_form, MPI_COMM_WORLD, myid, num_procs);
    end = clock();
    create_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    if (myid == 0) 
    {
        std::cout << "System Creation time: " << create_time << " seconds" << std::endl;
    }

    // Print the matrix to a file based on the matrix form.
    if (g_print_matrix)
    {
        std::string filename;
        switch (g_matrix_form)
        {
            case MatrixForm::SaddlePoint:
                filename = "matrix_saddle_point_proc" + std::to_string(myid) + ".dat";
                break;
            case MatrixForm::WeaklyCoupled:
                filename = "matrix_weakly_coupled_proc" + std::to_string(myid) + ".dat";
                break;
            case MatrixForm::StronglyCoupled:
                filename = "matrix_strongly_coupled_proc" + std::to_string(myid) + ".dat";
                break;
        }
        HYPRE_IJMatrixPrint(A, filename.c_str());
    }
    
    // Get the ParCSR and ParVector structures.
    HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
    HYPRE_IJVectorGetObject(b, (void**) &par_b);
    HYPRE_IJVectorGetObject(x, (void**) &par_x);
    

    // Create a BoomerAMG solver.
    HYPRE_Solver solver;
    HYPRE_BoomerAMGCreate(&solver);
    
    // Configure the solver.
    HYPRE_BoomerAMGSetMaxIter(solver, 500);        ///< max iterations as a solver.
    HYPRE_BoomerAMGSetTol(solver, 1e-7);           ///< convergence tolerance.
    HYPRE_BoomerAMGSetPrintLevel(solver, 3);        ///< more detailed output.
    if (g_solver_type == SolverType::UnknownBased)
    {
        // For weakly coupled PDEs.
        HYPRE_BoomerAMGSetCoarsenType(solver, 10);      ///< HMIS coarsening.
        HYPRE_BoomerAMGSetInterpType(solver, 6);        ///< extended+i interpolation.
        HYPRE_BoomerAMGSetPMaxElmts(solver, 4);         ///< max elements per row for interp.
        HYPRE_BoomerAMGSetAggNumLevels(solver, 0);      ///< no Aggressive coarsening.
        // HYPRE_BoomerAMGSetRelaxType(solver, 16);        ///< relaxation type.
        HYPRE_BoomerAMGSetNumSweeps(solver, 2);         ///< sweeps on each level.
    }

    if (g_solver_type == SolverType::NodeBased)
    {
        // For strongly coupled PDEs.
        HYPRE_BoomerAMGSetCoarsenType(solver, 10);      ///< HMIS coarsening.
        HYPRE_BoomerAMGSetInterpType(solver, 10);       ///< classical block interpolation.
        //HYPRE_BoomerAMGSetRelaxType(solver, 16);        ///< Chebyshev relaxation.
        HYPRE_BoomerAMGSetNumSweeps(solver, 2);         ///< sweeps on each level.
        HYPRE_BoomerAMGSetNumFunctions(solver, g_pde_size); ///< num of variables per grid point.
        HYPRE_BoomerAMGSetNodal(solver, 1);             ///< Nodal systems approach.
    }

    // Setup the solver.
    if (myid == 0)
        std::cout << "Begin BoomerAMG Setup." << std::endl;
    start = clock();
    HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
    end = clock();
    setup_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    if (myid == 0)
        std::cout << "BoomerAMG Setup time: " << setup_time << " seconds" << std::endl;
    
    // Solve the system.
    if (myid == 0)
        std::cout << "Begin BoomerAMG Solve." << std::endl;
    start = clock();
    HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
    end = clock();
    solve_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Get convergence info.
    double final_res_norm;
    int num_iterations;
    HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
    HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
    
    // Summary.
    if (myid == 0) 
    {
        switch (g_matrix_form)
        {
            case MatrixForm::SaddlePoint:
                std::cout << "The system is in saddle point form." << std::endl;
                break;
            case MatrixForm::WeaklyCoupled:
                std::cout << "The system is in weakly coupled form." << std::endl;
                break;
            case MatrixForm::StronglyCoupled:
                std::cout << "The system is in strongly coupled form." << std::endl;
                break;
        }

        switch (g_solver_type)
        {
            case SolverType::UnknownBased:
                std::cout << "The solver is unknown based." << std::endl;
                break;
            case SolverType::NodeBased:
                std::cout << "The solver is node based." << std::endl;
                break;
        }
        std::cout << "BoomerAMG Solve time: " << solve_time << " seconds" << std::endl;
        std::cout << "Number of iterations: " << num_iterations << std::endl;
        std::cout << "Final relative residual norm: " << final_res_norm << std::endl;
    }
    
    // Clean up.
    HYPRE_BoomerAMGDestroy(solver);
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(b);
    HYPRE_IJVectorDestroy(x);
    
    // Finalize Hypre.
    HYPRE_Finalize();
    
    // Finalize MPI.
    MPI_Finalize();
    
    return 0;
}