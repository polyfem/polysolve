#ifdef POLYSOLVE_WITH_HYPRE

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolverHypreGMRES.hpp>

#include <HYPRE_krylov.h>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverHypreGMRES::LinearSolverHypreGMRES()
    {
#ifdef MPI_VERSION
        /* Initialize MPI */
        int argc = 1;
        char name[] = "";
        char *argv[] = {name};
        char **argvv = &argv[0];
        int myid, num_procs;
        MPI_Init(&argc, &argvv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif
    }

    // Set solver parameters
    void LinearSolverHypreGMRES::setParameters(const json &params)
    {
        if (params.count("max_iter"))
        {
            max_iter_ = params["max_iter"];
        }
        if (params.count("pre_max_iter"))
        {
            pre_max_iter_ = params["pre_max_iter"];
        }

        if (params.count("conv_tol"))
        {
            conv_tol_ = params["conv_tol"];
        }
        else if (params.count("tolerance"))
        {
            conv_tol_ = params["tolerance"];
        }
    }

    void LinearSolverHypreGMRES::getInfo(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverHypreGMRES::analyzePattern(const StiffnessMatrix &Ain, const int precond_num)
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }

        has_matrix_ = true;
        const HYPRE_Int rows = Ain.rows();
        const HYPRE_Int cols = Ain.cols();

        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, rows - 1, 0, cols - 1, &A);
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

    void LinearSolverHypreGMRES::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        HYPRE_IJVector b;
        HYPRE_ParVector par_b;
        HYPRE_IJVector x;
        HYPRE_ParVector par_x;

        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, rhs.size() - 1, &b);
        HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(b);

        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, rhs.size() - 1, &x);
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

        /* GMRES */

        /* Create solver */
        HYPRE_Solver solver, eu;
        HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);

        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_ParCSRGMRESSetMaxIter(solver, max_iter_); /* max iterations */
        HYPRE_ParCSRGMRESSetTol(solver, conv_tol_);     /* conv. tolerance */
        HYPRE_ParCSRGMRESSetStopCrit(solver, 1);
        // HYPRE_ParCSRGMRESSetPrintLevel(solver, 2); /* print solve info */
        HYPRE_ParCSRGMRESSetLogging(solver, 1); /* needed to get run info later */

        /*HYPRE_EuclidCreate(MPI_COMM_WORLD, &eu);
        HYPRE_ParCSRGMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn)HYPRE_EuclidSolve, (HYPRE_PtrToSolverFcn)HYPRE_EuclidSetup, eu);*/

        /* Now setup and solve! */
        HYPRE_ParCSRGMRESSetup(solver, parcsr_A, par_b, par_x);
        HYPRE_ParCSRGMRESSolve(solver, parcsr_A, par_b, par_x);

        /* Run info - needed logging turned on */
        HYPRE_ParCSRGMRESGetNumIterations(solver, &num_iterations);
        HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);

        printf("\n");
        printf("Iterations = %lld\n", num_iterations);
        printf("Final Relative Residual Norm = %g\n", final_res_norm);
        printf("\n");

        /* Destroy solver and preconditioner */
        HYPRE_ParCSRGMRESDestroy(solver);

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

    LinearSolverHypreGMRES::~LinearSolverHypreGMRES()
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }
    }

} // namespace polysolve

#endif
