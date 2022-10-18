#ifdef POLYSOLVE_WITH_CUSOLVER

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolverCuSolverDN.cuh>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{
    LinearSolverCuSolverDN::LinearSolverCuSolverDN()
    {
        init();
        //TODO: fill this in -MP
    }

    LinearSolverCuSolverDN::init()
    {
        cusolverDnCreate(&cuHandle);
        //TODO: fill this in -MP
    }

    LinearSolverCuSolverDN::setParameters(const json &params)
    {
        cusolverDnCreateParams(&params);
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverCuSolverDN::factorize(const StiffnessMatrix &A)
    {
        int numrows = (int)A.rows();
        /*
        cusolverDnXgetrf(cuHandle, params, numrows, numrows, traits<double>::cuda_data_type, &A, 
        numrows, nullptr, traits<double>::cuda_data_type, )
        */
    }

    LinearSolverCuSolverDN::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        /*
        cusolverDnXgetrs(cuHandle, params, CUBLAS_OP_N,)
        */
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverCuSolverDN::~LinearSolverCuSolverDN()
    {
        cusolverDnDestroy(&cuHandle);
        //TODO: fill this in -MP
    }
    
}//namespace polysolve

#endif