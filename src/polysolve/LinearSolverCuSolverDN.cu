#ifdef POLYSOLVE_WITH_CUSOLVER

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolverCuSolverDN.cuh>
#include <fstream>
#include <string>
#include <iostream>

inline void gpuErrchk(cudaError_t code) {
  if (code != cudaSuccess) {
    throw cudaGetErrorString(code);
  }
}
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{
    LinearSolverCuSolverDN::LinearSolverCuSolverDN()
    {
        init();
    }

    void LinearSolverCuSolverDN::init()
    {
        cusolverDnCreate(&cuHandle);
        cusolverDnCreateParams(&cuParams);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cusolverDnSetStream(cuHandle, stream);
    }

    void LinearSolverCuSolverDN::setParameters(const json &params)
    {

    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverCuSolverDN::getInfo(json &params) const
    {

    }

    void LinearSolverCuSolverDN::analyzePattern(const StiffnessMatrix &A, const int precond_num)
    {

    }

    void LinearSolverCuSolverDN::analyzePattern(const Eigen::MatrixXd &A, const int precond_num)
    {

    }

    void LinearSolverCuSolverDN::factorize(const StiffnessMatrix &A)
    {
        numrows = (int)A.rows();
        Adense = Eigen::MatrixXd(A);
        factorize(Adense);
    }

    void LinearSolverCuSolverDN::factorize(const Eigen::MatrixXd &A)
    {
        numrows = (int)A.rows();

        //copy A to device
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
        gpuErrchk(cudaMemcpy(d_A, (const void *)A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice));
        
        cusolverDnXgetrf_bufferSize(cuHandle, cuParams, numrows, numrows, CUDA_R_64F, d_A,
                                    numrows, CUDA_R_64F, &d_lwork, &h_lwork);
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * d_lwork));
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&h_work), sizeof(double) * h_lwork));
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * numrows));
        int info = 0;
        
        //factorize
        cusolverStatus_t solvererr = cusolverDnXgetrf(cuHandle, cuParams, numrows, numrows, CUDA_R_64F, d_A, 
        numrows, d_Ipiv, CUDA_R_64F, d_work, d_lwork, h_work, h_lwork, d_info);

        if(solvererr == CUSOLVER_STATUS_INVALID_VALUE){
            throw std::invalid_argument("CUDA returned invalid value");
        }

        gpuErrchk(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    }

    void LinearSolverCuSolverDN::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        //copy b to device
        gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * b.size()));
        gpuErrchk(cudaMemcpy(d_b, (const void *)b.data(), sizeof(double) * b.size(), cudaMemcpyHostToDevice));

        //solve
        cusolverStatus_t solvererr = cusolverDnXgetrs(cuHandle, cuParams, CUBLAS_OP_N, numrows, 1,
            CUDA_R_64F, d_A, numrows, d_Ipiv,
            CUDA_R_64F, d_b, numrows, d_info);
        if(solvererr == CUSOLVER_STATUS_INVALID_VALUE){
            throw std::invalid_argument("CUDA returned invalid value");
        }
        int info = 0;
        gpuErrchk(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

        //copy result to x
        gpuErrchk(cudaMemcpy(x.data(), d_b, sizeof(double) * x.size(), cudaMemcpyDeviceToHost));
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverCuSolverDN::~LinearSolverCuSolverDN()
    {
        gpuErrchk(cudaFree(d_A));
        gpuErrchk(cudaFree(d_b));
        gpuErrchk(cudaFree(d_work));
        gpuErrchk(cudaFree(d_Ipiv));
        gpuErrchk(cudaFree(d_info));

        cusolverDnDestroyParams(cuParams);
        cusolverDnDestroy(cuHandle);
        cudaStreamDestroy(stream);
        cudaDeviceReset();
    }
    
}//namespace polysolve

#endif