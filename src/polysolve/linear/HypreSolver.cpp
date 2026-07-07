#ifdef POLYSOLVE_WITH_HYPRE

////////////////////////////////////////////////////////////////////////////////
#include "HypreSolver.hpp"

#include <HYPRE_krylov.h>
#include <HYPRE_utilities.h>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve::linear
{

    ////////////////////////////////////////////////////////////////////////////////

    HypreSolver::HypreSolver()
    {
        precond_num_ = 0;
#ifdef HYPRE_ENABLE_MPI
        int done_already;

        MPI_Initialized(&done_already);
        if (!done_already)
        {
            /* Initialize MPI */
            int argc = 1;
            char name[] = "";
            char *argv[] = {name};
            char **argvv = &argv[0];
            int myid, num_procs;
            MPI_Init(&argc, &argvv);
            MPI_Comm_rank(MPI_COMM_WORLD, &myid);
            MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        }
#endif
        if (!HYPRE_Initialized())
        {
            HYPRE_Initialize();
        }
    }

    // Set solver parameters
    void HypreSolver::set_parameters(const json &params)
    {
        if (params.contains("Hypre"))
        {
            if (params["Hypre"].contains("max_iter"))
            {
                max_iter_ = params["Hypre"]["max_iter"];
            }
            if (params["Hypre"].contains("pre_max_iter"))
            {
                pre_max_iter_ = params["Hypre"]["pre_max_iter"];
            }
            if (params["Hypre"].contains("tolerance"))
            {
                conv_tol_ = params["Hypre"]["tolerance"];
            }
            if (params["Hypre"].contains("theta"))
            {
                theta = params["Hypre"]["theta"];
            }
            if (params["Hypre"].contains("nodal_coarsening"))
            {
                nodal_coarsening = params["Hypre"]["nodal_coarsening"];
            }
            if (params["Hypre"].contains("interp_rbms"))
            {
                interp_rbms = params["Hypre"]["interp_rbms"];
            }
            if (params["Hypre"].contains("dimension"))
            {
                dimension_ = params["Hypre"]["dimension"];
            }
        }
    }

    void HypreSolver::get_info(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void HypreSolver::factorize(const StiffnessMatrix &Ain)
    {
        assert(precond_num_ > 0);

        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }

        has_matrix_ = true;
        const HYPRE_Int rows = Ain.rows();
        const HYPRE_Int cols = Ain.cols();
#ifdef HYPRE_ENABLE_MPI
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, rows - 1, 0, cols - 1, &A);
#else
        HYPRE_IJMatrixCreate(0, 0, rows - 1, 0, cols - 1, &A);
#endif
        // HYPRE_IJMatrixSetPrintLevel(A, 2);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        // HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, cols, values);

        // TODO: More efficient initialization of the Hypre matrix?
        for (HYPRE_Int k = 0; k < Ain.outerSize(); ++k)
        {
            for (StiffnessMatrix::InnerIterator it(Ain, k); it; ++it)
            {
                HYPRE_Int row[1]; 
                int counter = 0;
                std::vector<HYPRE_Int> cols;
                std::vector<double> vals;
                for (StiffnessMatrix::InnerIterator it(Ain, k); it; ++it)
                {
                    // since A is symmetric, we swap rows and columns for more efficient initialization
                    ++counter;
                    row[0] = it.col();
                    cols.push_back((HYPRE_Int)it.row());
                    vals.push_back(it.value());
                }
                HYPRE_Int n_cols[1] = {counter};
                HYPRE_IJMatrixSetValues(A, 1, n_cols, row, cols.data(), vals.data());
            }
        }

        HYPRE_IJMatrixAssemble(A);
        HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);
    }

    ////////////////////////////////////////////////////////////////////////////////

    namespace
    {

        void eigen_to_hypre_par_vec(HYPRE_ParVector &par_x, HYPRE_IJVector &ij_x, const Eigen::VectorXd &x)
        {
            HYPRE_IJVectorCreate(0, 0, x.size() - 1, &ij_x);
            HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize(ij_x);

            HYPRE_IJVectorSetValues(ij_x, x.size(), nullptr, x.data());

            HYPRE_IJVectorAssemble(ij_x);
            HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);
        }

        void hypre_vec_to_eigen(const HYPRE_IJVector &ij_x, Eigen::Ref<Eigen::VectorXd> &x)
        {
            HYPRE_IJVectorGetValues(ij_x, x.size(), nullptr, x.data());
        }

        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 10; // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
            int agg_levels = 1;    // number of aggressive coarsening levels
            double theta = 0.5;   // strength threshold: 0.25, 0.5, 0.8

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i, 0 = classical
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 8;   // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
            int relax_sweeps = 1; // relaxation sweeps on each level

            // Additional options:
            int print_level = 0; // print AMG iterations? 1 = no, 2 = yes
            int max_levels = 25; // max number of levels in AMG hierarchy

            HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
            HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
            HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
            HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
            HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
            HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
            HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);

            // Use as a preconditioner (one V-cycle, zero tolerance)
            HYPRE_BoomerAMGSetMaxIter(amg_precond, 1);
            HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
        }

        void HypreBoomerAMG_SetElasticityOptions(HYPRE_Solver &amg_precond, int dim, double theta, bool nodal_coarsening, bool interp_rbms, const Eigen::MatrixXd &positions, std::vector<HYPRE_IJVector> &rbms, std::vector<HYPRE_ParVector> &par_rbms)
        {
            // Make sure the systems AMG options are set
            HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

            // More robust options with respect to convergence
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);

            // Nodal coarsening options (nodal coarsening is required for this solver)
            // See hypre's new_ij driver and the paper for descriptions.
            int nodal = 4;        // strength reduction norm: 1, 3 or 4
            int nodal_diag = 1;   // diagonal in strength matrix: 0, 1 or 2
            int relax_coarse = 8; // smoother on the coarsest grid: 8, 99 or 29

            // Elasticity interpolation options
            int interp_vec_variant = 2;    // 1 = GM-1, 2 = GM-2, 3 = LN
            int q_max = 4;                 // max elements per row for each Q
            int smooth_interp_vectors = 1; // smooth the rigid-body modes?

            // Optionally pre-process the interpolation matrix through iterative weight
            // refinement (this is generally applicable for any system)
            int interp_refine = 1;

            if (nodal_coarsening) 
            {
                HYPRE_BoomerAMGSetNodal(amg_precond, nodal);
                HYPRE_BoomerAMGSetNodalDiag(amg_precond, nodal_diag);
                HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_coarse, 3);
            }

            if (interp_rbms)
            {
                if (dim != 2 && dim != 3)
                {
                    assert(false);
                }

                HYPRE_BoomerAMGSetInterpVecVariant(amg_precond, interp_vec_variant);
                HYPRE_BoomerAMGSetInterpVecQMax(amg_precond, q_max);

                // HYPRE_BoomerAMGSetSmoothInterpVectors(amg_precond, smooth_interp_vectors);
                // HYPRE_BoomerAMGSetInterpRefine(amg_precond, interp_refine);

                Eigen::VectorXd rbm_xy, rbm_zx, rbm_yz;
                rbm_xy.resize(positions.size());
                rbm_xy.setZero();

                if (dim == 3)
                {
                    rbm_zx.resize(positions.size());
                    rbm_yz.resize(positions.size());

                    rbm_zx.setZero();
                    rbm_yz.setZero();
                }

                for (int i = 0; i < positions.rows(); ++i)
                {
                    rbm_xy(0 + i*dim) = positions(i, 1);
                    rbm_xy(1 + i*dim) = -1 * positions(i, 0);

                    if (dim == 3)
                    {
                        rbm_zx(1 + i*dim) = positions(i, 2);
                        rbm_zx(2 + i*dim) = -1 * positions(i, 1);

                        rbm_yz(2 + i*dim) = positions(i, 0);
                        rbm_yz(0 + i*dim) = -1 * positions(i, 2);
                    }
                }

                eigen_to_hypre_par_vec(par_rbms[0], rbms[0], rbm_xy);
                if (dim == 3)
                {
                    eigen_to_hypre_par_vec(par_rbms[1], rbms[1], rbm_zx);
                    eigen_to_hypre_par_vec(par_rbms[2], rbms[2], rbm_yz);
                }

                HYPRE_BoomerAMGSetInterpVectors(amg_precond, par_rbms.size(), &(par_rbms[0]));
            }
        }

    } // anonymous namespace

    ////////////////////////////////////////////////////////////////////////////////

    void HypreSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        HYPRE_IJVector b;
        HYPRE_ParVector par_b;
        HYPRE_IJVector x;
        HYPRE_ParVector par_x;

        eigen_to_hypre_par_vec(par_b, b, rhs);
        eigen_to_hypre_par_vec(par_x, x, result);

        /* PCG with AMG preconditioner */

        /* Create solver */
        HYPRE_Solver solver, precond;
#ifdef HYPRE_ENABLE_MPI
        HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
#else
        HYPRE_ParCSRPCGCreate(0, &solver);
#endif

        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_PCGSetMaxIter(solver, max_iter_); /* max iterations */
        HYPRE_PCGSetTol(solver, conv_tol_);     /* conv. tolerance */
        HYPRE_PCGSetTwoNorm(solver, 1);         /* use the two norm as the stopping criteria */
        // HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
        HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

        /* Now set up the AMG preconditioner and specify any parameters */
        HYPRE_BoomerAMGCreate(&precond);

        const int num_rbms = dimension_ == 2 ? 1 : 3;
        std::vector<HYPRE_ParVector> par_rbms(num_rbms);
        std::vector<HYPRE_IJVector> rbms(num_rbms);
        HypreBoomerAMG_SetDefaultOptions(precond);
        if (dimension_ > 1)
        {
            Eigen::MatrixXd positions_;
            assert(!interp_rbms);
            HypreBoomerAMG_SetElasticityOptions(precond, dimension_, theta, nodal_coarsening, interp_rbms, positions_, rbms, par_rbms);
        }

        /* Set the PCG preconditioner */
        HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond);

        /* Now setup and solve! */
        HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
        HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);

        /* Run info - needed logging turned on */
        HYPRE_PCGGetNumIterations(solver, &num_iterations);
        HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

        // printf("\n");
        // printf("Iterations = %lld\n", num_iterations);
        // printf("Final Relative Residual Norm = %g\n", final_res_norm);
        // printf("\n");

        /* Destroy solver and preconditioner */
        HYPRE_BoomerAMGDestroy(precond);
        HYPRE_ParCSRPCGDestroy(solver);

        assert(result.size() == rhs.size());
        hypre_vec_to_eigen(x, result);

        HYPRE_IJVectorDestroy(b);
        HYPRE_IJVectorDestroy(x);
    }

    ////////////////////////////////////////////////////////////////////////////////

    HypreSolver::~HypreSolver()
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }
    }

} // namespace polysolve::linear

#endif
