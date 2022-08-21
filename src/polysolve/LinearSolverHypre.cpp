#ifdef POLYSOLVE_WITH_HYPRE

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolverHypre.hpp>

#include <HYPRE_krylov.h>
#include <HYPRE_utilities.h>
#include <_hypre_utilities.h>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverHypre::LinearSolverHypre()
    {
        precond_num_ = 0;
#ifdef HYPRE_WITH_MPI
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
        HYPRE_Init();
    }

    // Set solver parameters
    void LinearSolverHypre::setParameters(const json &params)
    {
        if (params.contains("Hypre"))
        {
            if (params["Hypre"].contains("block_size"))
            {
                if (params["Hypre"]["block_size"] == 2 || params["Hypre"]["block_size"] == 3)
                {
                    dimension_ = params["Hypre"]["block_size"];
                }
            }
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
        }
    }

    void LinearSolverHypre::getInfo(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverHypre::factorize(const StiffnessMatrix &Ain)
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
#ifdef HYPRE_WITH_MPI
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, rows - 1, 0, cols - 1, &A);
#else
        HYPRE_IJMatrixCreate(hypre_MPI_COMM_WORLD, 0, rows - 1, 0, cols - 1, &A);
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
                const HYPRE_Int i[1] = {it.row()};
                const HYPRE_Int j[1] = {it.col()};
                const HYPRE_Complex v[1] = {it.value()};
                HYPRE_Int n_cols[1] = {1};

                HYPRE_IJMatrixSetValues(A, 1, n_cols, i, j, v);
            }
        }

        HYPRE_IJMatrixAssemble(A);
        HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);
    }

    ////////////////////////////////////////////////////////////////////////////////

    namespace
    {

        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 10; // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
            int agg_levels = 1;    // number of aggressive coarsening levels
            double theta = 0.25;   // strength threshold: 0.25, 0.5, 0.8

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i, 0 = classical
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 16;  // 8 = l1-GS, 6 = symm. GS, 3 = GS, 16 = Chebyshev, 18 = l1-Jacobi
            int relax_sweeps = 1; // relaxation sweeps on each level

            // Additional options:
            int print_level = 0; // print AMG iterations? 1 = no, 2 = yes
            int max_levels = 6;  // max number of levels in AMG hierarchy

            // Chebyshev Settings
            int eig_est = 100;   // Number of CG iterations to determine the smallest and largest eigenvalue
            double ratio = 0.008333333333;

            HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
            HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
            HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
            HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
            HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
            HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
            HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);

            // To do, figure out what does these functions mean
            // Defines the Order for Chebyshev smoother. The default is 2 (valid options are 1-4).
            HYPRE_BoomerAMGSetChebyOrder(amg_precond, 4);
            // Fraction of the spectrum to use for the Chebyshev smoother. The default is .3 (i.e., damp on upper 30% of the spectrum).
            HYPRE_BoomerAMGSetChebyFraction(amg_precond, ratio);

            // Settings from AMGCL
            HYPRE_BoomerAMGSetChebyScale(amg_precond, 1);
            HYPRE_BoomerAMGSetChebyVariant(amg_precond, 0);
            HYPRE_BoomerAMGSetChebyEigEst(amg_precond, eig_est);

            // Use as a preconditioner (one V-cycle, zero tolerance)
            HYPRE_BoomerAMGSetMaxIter(amg_precond, 1);
            HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
        }

        void HypreBoomerAMG_SetElasticityOptions(HYPRE_Solver &amg_precond, int dim)
        {
            // Make sure the systems AMG options are set
            HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

            // More robust options with respect to convergence
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
            if (dim > 2)
            {
                // Hypre recommend setting 0.5 for 3D problem, 0.25 for 2D and scalar problem
                HYPRE_BoomerAMGSetStrongThreshold(amg_precond, 0.5);
            }
            // Nodal coarsening options (nodal coarsening is required for this solver)
            // See hypre's new_ij driver and the paper for descriptions.
            int nodal = 4;      // strength reduction norm: 1, 3 or 4
            int nodal_diag = 1; // diagonal in strength matrix: 0, 1 or 2
            // int relax_coarse = 8; // smoother on the coarsest grid: 8, 99 or 29

            // Chebyshev preconditioner
            int relax_coarse = 16; // Chebyshev relax

            // Elasticity interpolation options
            int interp_vec_variant = 2;    // 1 = GM-1, 2 = GM-2, 3 = LN
            int q_max = 4;                 // max elements per row for each Q
            int smooth_interp_vectors = 1; // smooth the rigid-body modes?

            // Optionally pre-process the interpolation matrix through iterative weight
            // refinement (this is generally applicable for any system)
            int interp_refine = 1;

            HYPRE_BoomerAMGSetNodal(amg_precond, nodal);
            HYPRE_BoomerAMGSetNodalDiag(amg_precond, nodal_diag);
            HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_coarse, 3);

            HYPRE_BoomerAMGSetInterpVecVariant(amg_precond, interp_vec_variant);
            HYPRE_BoomerAMGSetInterpVecQMax(amg_precond, q_max);
            // HYPRE_BoomerAMGSetSmoothInterpVectors(amg_precond, smooth_interp_vectors);
            // HYPRE_BoomerAMGSetInterpRefine(amg_precond, interp_refine);

            // RecomputeRBMs();
            // HYPRE_BoomerAMGSetInterpVectors(amg_precond, rbms.Size(), rbms.GetData());
        }

    } // anonymous namespace

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverHypre::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        HYPRE_IJVector b;
        HYPRE_ParVector par_b;
        HYPRE_IJVector x;
        HYPRE_ParVector par_x;

#ifdef HYPRE_WITH_MPI
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, rhs.size() - 1, &b);
#else
        HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, 0, rhs.size() - 1, &b);
#endif
        HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(b);
#ifdef HYPRE_WITH_MPI
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, rhs.size() - 1, &x);
#else
        HYPRE_IJVectorCreate(hypre_MPI_COMM_WORLD, 0, rhs.size() - 1, &x);
#endif
        HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(x);

        assert(result.size() == rhs.size());

        for (HYPRE_Int i = 0; i < rhs.size(); ++i)
        {
            const HYPRE_Int index[1] = {i};
            const HYPRE_Complex v[1] = {HYPRE_Complex(rhs(i))};
            const HYPRE_Complex z[1] = {HYPRE_Complex(result(i))};

            HYPRE_IJVectorSetValues(b, 1, index, v);
            HYPRE_IJVectorSetValues(x, 1, index, z);
        }

        HYPRE_IJVectorAssemble(b);
        HYPRE_IJVectorGetObject(b, (void **)&par_b);

        HYPRE_IJVectorAssemble(x);
        HYPRE_IJVectorGetObject(x, (void **)&par_x);

        /* PCG with AMG preconditioner */

        /* Create solver */
        HYPRE_Solver solver, precond;
#ifdef HYPRE_WITH_MPI
        HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
#else
        HYPRE_ParCSRPCGCreate(hypre_MPI_COMM_WORLD, &solver);
#endif

        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_PCGSetMaxIter(solver, max_iter_); /* max iterations */
        HYPRE_PCGSetTol(solver, conv_tol_);     /* conv. tolerance */
        HYPRE_PCGSetTwoNorm(solver, 1);         /* use the two norm as the stopping criteria */
        // HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
        HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

        /* Now set up the AMG preconditioner and specify any parameters */
        HYPRE_BoomerAMGCreate(&precond);

#if 0
    //HYPRE_BoomerAMGSetPrintLevel(precond, 2); /* print amg solution info */
    HYPRE_BoomerAMGSetCoarsenType(precond, 6);
    HYPRE_BoomerAMGSetOldDefault(precond);
    HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
    HYPRE_BoomerAMGSetNumSweeps(precond, 1);
    HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
    HYPRE_BoomerAMGSetMaxIter(precond, pre_max_iter_); /* do only one iteration! */
#endif

        HypreBoomerAMG_SetDefaultOptions(precond);
        if (dimension_ > 1)
        {
            HypreBoomerAMG_SetElasticityOptions(precond, dimension_);
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
        for (HYPRE_Int i = 0; i < rhs.size(); ++i)
        {
            const HYPRE_Int index[1] = {i};
            HYPRE_Complex v[1];
            HYPRE_IJVectorGetValues(x, 1, index, v);

            result(i) = v[0];
        }

        HYPRE_IJVectorDestroy(b);
        HYPRE_IJVectorDestroy(x);
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverHypre::~LinearSolverHypre()
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }
    }

} // namespace polysolve

#endif