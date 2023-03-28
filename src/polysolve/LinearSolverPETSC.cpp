#ifdef POLYSOLVE_WITH_PETSC

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolverPETSC.hpp>
#include <fstream>
#include <string>
#include <iostream>

namespace polysolve
{
    LinearSolverPETSC::LinearSolverPETSC()
    {
        init();
    }

    int LinearSolverPETSC::init()
    {
        PetscCall(PetscInitialize(NULL, NULL, NULL, NULL));
        PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
        return 0;
    }

    void LinearSolverPETSC::setParameters(const json &params)
    {
    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverPETSC::getInfo(json &params) const
    {
    }

    void LinearSolverPETSC::analyzePattern(const StiffnessMatrix &A, const int precond_num)
    {
    }

    void LinearSolverPETSC::analyzePattern(const Eigen::MatrixXd &A, const int precond_num)
    {
    }

    void LinearSolverPETSC::factorize(StiffnessMatrix &A, int AIJ_CUSPARSE, int SOLVER_INDEX)
    {
        GPU_vec = AIJ_CUSPARSE;
        /*CHOLMOD requires 64-bit int indices for GPU backend support*/
#ifdef CHOLMOD_WITH_GPU
        std::vector<long int> outer_(A.outerSize() + 1);
        std::vector<long int> inner_(A.nonZeros());
        for (int k = 0; k < A.outerSize() + 1; ++k)
        {
            outer_[k] = A.outerIndexPtr()[k];
        }
        for (int j = 0; j < A.nonZeros(); ++j)
        {
            inner_[j] = A.innerIndexPtr()[j];
        }
        MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, A.rows(), A.cols(), outer_.data(), inner_.data(), A.valuePtr(), &A_petsc);
        MatConvert(A_petsc, MATAIJCUSPARSE, MAT_INPLACE_MATRIX, &A_petsc);
#else
        MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, A.rows(), A.cols(), A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), &A_petsc);
        if (AIJ_CUSPARSE)
            MatConvert(A_petsc, MATAIJCUSPARSE, MAT_INPLACE_MATRIX, &A_petsc);
#endif

        // IF EIGEN MATRIX IS ROW MAJOR WE DO A TRANSPOSE
        // MatTranspose(A_petsc, MAT_INPLACE_MATRIX, &A_petsc);

        MatCreateVecs(A_petsc, &x_petsc, NULL);

        KSPCreate(PETSC_COMM_WORLD, &ksp);
        KSPSetOperators(ksp, A_petsc, A_petsc);
        if (SOLVER_INDEX == 5)
        {
            KSPSetType(ksp, KSPGMRES);
            KSPSetTolerances(ksp, PETSC_DEFAULT, 1e-50, PETSC_DEFAULT, PETSC_DEFAULT);
        }
        else
            KSPSetType(ksp, KSPPREONLY);

        KSPGetPC(ksp, &pc);

        switch (SOLVER_INDEX)
        {
        case 0:
            PCSetType(pc, PCLU);
            PCFactorSetMatSolverType(pc, MATSOLVERMKL_PARDISO);
            break;
        case 1:
            PCSetType(pc, PCLU);
            PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST);
            break;
        case 2:
            PCSetType(pc, PCCHOLESKY);
            PCFactorSetMatSolverType(pc, MATSOLVERCHOLMOD);
            break;
        case 3:
            PCSetType(pc, PCLU);
            PCFactorSetMatSolverType(pc, MATSOLVERMUMPS);
            break;
        case 4:
            PCSetType(pc, PCLU);
            PCFactorSetMatSolverType(pc, MATSOLVERCUSPARSE);
            break;
        case 5:
            PCSetType(pc, PCLU);
            PCFactorSetMatSolverType(pc, MATSOLVERSTRUMPACK);
            break;
            //        case 6:
            // TODO : FIX HYPRE PARAMETERS
            //            PCSetType(pc, PCHYPRE);
            //            PCHYPRESetType(pc, "boomeramg");
            //            break;
        default:
            PCSetType(pc, PCLU);
            PCFactorSetMatSolverType(pc, MATSOLVERPETSC);
        }

        PCFactorSetUpMatSolverType(pc); /* call MatGetFactor() to create F */
        PCFactorGetMatrix(pc, &F);

        /*Parameters obtained from https://petsc.org/release/src/ksp/ksp/tutorials/ex52.c.html*/
        if (SOLVER_INDEX == 3)
        {
            MatMumpsSetIcntl(F, 7, 2);
            /* threshold for row pivot detection */
            MatMumpsSetIcntl(F, 24, 1);
            MatMumpsSetCntl(F, 3, 1.e-6);
        }
        if (SOLVER_INDEX == 1)
        {
            MatSuperluSetILUDropTol(F, 1.e-8);
        }
        if (SOLVER_INDEX == 5)
        {
            /* Set the fill-reducing reordering.                              */
            MatSTRUMPACKSetReordering(F, MAT_STRUMPACK_METIS);
            /* Since this is a simple discretization, the diagonal is always  */
            /* nonzero, and there is no need for the extra MC64 permutation.  */
            MatSTRUMPACKSetColPerm(F, PETSC_FALSE);
            /* The compression tolerance used when doing low-rank compression */
            /* in the preconditioner. This is problem specific!               */
            // MatSTRUMPACKSetHSSRelTol(F, 1.e-3);
            /* Set minimum matrix size for HSS compression to 15 in order to  */
            /* demonstrate preconditioner on small problems. For performance  */
            /* a value of say 500 is better.                                  */
            MatSTRUMPACKSetHSSMinSepSize(F, 500);
            /* You can further limit the fill in the preconditioner by        */
            /* setting a maximum rank                                         */
            MatSTRUMPACKSetHSSMaxRank(F, 100);
            /* Set the size of the diagonal blocks (the leafs) in the HSS     */
            /* approximation. The default value should be better for real     */
            /* problems. This is mostly for illustration on a small problem.  */
            //           MatSTRUMPACKSetHSSLeafSize(F, 4);
        }
        return;
    }

    void LinearSolverPETSC::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        if (GPU_vec)
            VecCreateSeqCUDAWithArrays(PETSC_COMM_WORLD, 1, b.rows() * b.cols(), b.data(), NULL, &b_petsc);
        else
            VecCreateSeqWithArray(PETSC_COMM_WORLD, 1, b.rows() * b.cols(), b.data(), &b_petsc);

        /*USEFUL FOR VARIABLE TEST CASES*/
        // KSPSetFromOptions(ksp);
        // KSPSetUp(ksp);
        KSPSolve(ksp, b_petsc, x_petsc);

        x.resize(b.rows() * b.cols(), 1);

        int pidx = 0;
        for (int i = 0; i < b.rows() * b.cols(); ++i)
        {
            PetscInt ix[] = {pidx};
            PetscScalar y[] = {0};
            VecGetValues(x_petsc, 1, ix, y);
            x(i, 0) = y[0];
            pidx++;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverPETSC::~LinearSolverPETSC()
    {
        KSPDestroy(&ksp);
        MatDestroy(&A_petsc);
        VecDestroy(&b_petsc);
        VecDestroy(&x_petsc);
        // PetscFinalize();
        // TODO: FIX THIS FOR POLYFEM
        //  MISSING: SET A EXTERNAL FINALIZE
    }

} // namespace polysolve

#endif