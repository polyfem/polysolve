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

        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cusolverDnSetStream(cuHandle, stream);
        //TODO: fill this in -MP
    }

    LinearSolverCuSolverDN::setParameters(const json &params)
    {
        cusolverDnCreateParams(&cuParams);

        //TODO: make a json option for legacy solver vs new solver
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverCuSolverDN::factorize(const StiffnessMatrix &A)
    {
        //find number of rows in A
        numrows = (int)A.rows();

        //copy A to device
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size());
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream);

        //calculate buffer size
        cusolverDnXgetrf_bufferSize(cuHandle, cuParams, numrows, numrows, traits<double>::cuda_data_type, d_A,
                                    numrows, traits<double>::cuda_data_type, &d_lwork, &h_lwork);
        
        //factorize
        cusolverDnXgetrf(cuHandle, cuParams, numrows, numrows, traits<double>::cuda_data_type, &A, 
        numrows, nullptr, traits<double>::cuda_data_type, d_work, d_lwork, h_work, h_lwork, nullptr);
    }

    LinearSolverCuSolverDN::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        //copy b to device
        cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * b.size());
        cudaMemcpyAsync(d_b, b.data(), sizeof(double) * b.size(), cudaMemcpyHostToDevice, stream);

        //solve
        cusolverDnXgetrs(cuHandle, cuParams, CUBLAS_OP_N, numrows, 1,
            traits<double>::cuda_data_type, d_A, numrows, nullptr,
            traits<double>::cuda_data_type, d_b, numrows, nullptr);
        
        //copy result to x
        cudaMemcpyAsync(x.data(), d_b, sizeof(data_type) * x.size(), cudaMemcpyDeviceToHost,
                               stream)
        //TODO: fill this in -MP
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverCuSolverDN::~LinearSolverCuSolverDN()
    {
        cusolverDnDestroyParams(cuParams);
        cusolverDnDestroy(cuHandle);
        cudaStreamDestroy(stream);
        //TODO: fill this in -MP
    }
    
}//namespace polysolve

#endif