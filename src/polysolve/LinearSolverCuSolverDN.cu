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

    void LinearSolverCuSolverDN::init()
    {
        cusolverDnCreate(&cuHandle);

        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cusolverDnSetStream(cuHandle, stream);
        //TODO: fill this in -MP
    }

    void LinearSolverCuSolverDN::setParameters(const json &params)
    {
        cusolverDnCreateParams(&cuParams);

        //TODO: make a json option for legacy solver vs new solver
    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverCuSolverDN::getInfo(json &params) const
    {

    }

    void LinearSolverCuSolverDN::analyzePattern(const StiffnessMatrix &A, const int precond_num)
    {

    }

    void LinearSolverCuSolverDN::factorize(const StiffnessMatrix &A)
    {
        //find number of rows in A
        numrows = (int)A.rows();

        //convert A to dense (temporary)
        Adense = Eigen::MatrixXd(A);

        //copy A to device
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * Adense.size());
        cudaMemcpyAsync(d_A, (const void *)Adense.data(), sizeof(double) * Adense.size(), cudaMemcpyHostToDevice, stream);

        //calculate buffer size
        cusolverDnXgetrf_bufferSize(cuHandle, cuParams, numrows, numrows, CUDA_R_64F, d_A,
                                    numrows, CUDA_R_64F, &d_lwork, &h_lwork);
        
        //factorize
        cusolverDnXgetrf(cuHandle, cuParams, numrows, numrows, CUDA_R_64F, &Adense, 
        numrows, nullptr, CUDA_R_64F, d_work, d_lwork, h_work, h_lwork, nullptr);
    }
    }

    void LinearSolverCuSolverDN::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        //copy b to device
        cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * b.size());
        cudaMemcpyAsync(d_b, (const void *)b.data(), sizeof(double) * b.size(), cudaMemcpyHostToDevice, stream);

        //solve
        cusolverDnXgetrs(cuHandle, cuParams, CUBLAS_OP_N, numrows, 1,
            CUDA_R_64F, d_A, numrows, nullptr,
            CUDA_R_64F, d_b, numrows, nullptr);
        
        //copy result to x
        cudaMemcpyAsync(x.data(), d_b, sizeof(double) * x.size(), cudaMemcpyDeviceToHost,
                               stream);
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