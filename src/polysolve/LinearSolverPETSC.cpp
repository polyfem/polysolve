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

    void LinearSolverPETSC::init()
    {
        //        PetscCall(PetscInitialize(NULL, NULL, NULL, NULL));
        //        PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
        PetscInitialize(NULL, NULL, NULL, NULL);
        PetscDeviceInitialize(PETSC_DEVICE_CUDA);
        return;
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

    void LinearSolverPETSC::factorize(StiffnessMatrix &A, int dummy)
    {
        //        PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, A.rows(), A.cols(), A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), &A_petsc));
        //        PetscCall(MatConvert(A_petsc, MATAIJCUSPARSE, MAT_INPLACE_MATRIX, &A_petsc));
        //        PetscCall(MatTranspose(A_petsc, MAT_INPLACE_MATRIX, &A_petsc));
        //        PetscCall(MatCreateVecs(A_petsc, &x_petsc, NULL));
        MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, A.rows(), A.cols(), A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), &A_petsc);
        //   MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, A.rows(), A.cols(), A_outer, A_inner, A_value, &A_petsc);
        MatConvert(A_petsc, MATAIJCUSPARSE, MAT_INPLACE_MATRIX, &A_petsc);
        MatTranspose(A_petsc, MAT_INPLACE_MATRIX, &A_petsc);
        MatCreateVecs(A_petsc, &x_petsc, NULL);
    }

    void LinearSolverPETSC::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        // copy b to device
        //        PetscCall(VecCreateSeqCUDAWithArrays(PETSC_COMM_SELF, 1, b.rows() * b.cols(), b.data(), NULL, &b_petsc));
        //
        //        PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
        //        PetscCall(KSPSetOperators(ksp, A_petsc, A_petsc));
        //        PetscCall(KSPSetTolerances(ksp, PETSC_DEFAULT, 1.e-50, PETSC_DEFAULT, PETSC_DEFAULT));
        //
        //        KSPSetType(ksp, KSPPREONLY);
        //        PetscCall(KSPGetPC(ksp, &pc));
        //        PCSetType(pc, PCLU);
        //        PCFactorSetMatSolverType(pc, MATSOLVERMUMPS);
        //
        //        PCFactorSetUpMatSolverType(pc); /* call MatGetFactor() to create F */
        //        PCFactorGetMatrix(pc, &F);
        //        MatMumpsSetCntl(F, 3, 1.e-6);
        //
        //        PetscCall(KSPSetFromOptions(ksp));
        //        KSPSetUp(ksp);
        //        PetscCall(KSPSolve(ksp, b_petsc, x_petsc));
        VecCreateSeqCUDAWithArrays(PETSC_COMM_SELF, 1, b.rows() * b.cols(), b.data(), NULL, &b_petsc);

        KSPCreate(PETSC_COMM_SELF, &ksp);
        KSPSetOperators(ksp, A_petsc, A_petsc);
        KSPSetTolerances(ksp, PETSC_DEFAULT, 1.e-50, PETSC_DEFAULT, PETSC_DEFAULT);
        //
        KSPSetType(ksp, KSPPREONLY);
        KSPGetPC(ksp, &pc);
        PCSetType(pc, PCLU);
        PCFactorSetMatSolverType(pc, MATSOLVERMUMPS);
        //
        PCFactorSetUpMatSolverType(pc); /* call MatGetFactor() to create F */
        PCFactorGetMatrix(pc, &F);
        MatMumpsSetCntl(F, 3, 1.e-6);
        //
        KSPSetFromOptions(ksp);
        KSPSetUp(ksp);
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
        //        PetscCall(KSPDestroy(&ksp));
        //        PetscCall(MatDestroy(&A_petsc));
        //        PetscCall(VecDestroy(&b_petsc));
        //        PetscCall(VecDestroy(&x_petsc));
        //        PetscCall(PetscFinalize());
        KSPDestroy(&ksp);
        MatDestroy(&A_petsc);
        VecDestroy(&b_petsc);
        VecDestroy(&x_petsc);
        PetscFinalize();
    }

} // namespace polysolve

#endif