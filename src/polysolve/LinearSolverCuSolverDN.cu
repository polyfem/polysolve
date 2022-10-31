#ifdef POLYSOLVE_WITH_CUSOLVER

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolverCuSolverDN.cuh>
#include <fstream>
#include <string>
#include <iostream>
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
        //TODO: make a json option for legacy solver vs new solver
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
        //find number of rows in A
        numrows = (int)A.rows();

        //convert A to dense
        Adense = Eigen::MatrixXd(A);

        //copy A to device
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * Adense.size());
        cudaMemcpy(d_A, (const void *)Adense.data(), sizeof(double) * Adense.size(), cudaMemcpyHostToDevice);
        
        //allocate pivots
        cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * numrows);

        //calculate buffer size
        cusolverDnXgetrf_bufferSize(cuHandle, cuParams, numrows, numrows, CUDA_R_64F, d_A,
                                    numrows, CUDA_R_64F, &d_lwork, &h_lwork);
        cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * d_lwork);
        cudaMalloc(reinterpret_cast<void **>(&h_work), sizeof(double) * h_lwork);
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
        
        //factorize
        cusolverStatus_t solvererr = cusolverDnXgetrf(cuHandle, cuParams, numrows, numrows, CUDA_R_64F, d_A, 
        numrows, d_Ipiv, CUDA_R_64F, d_work, d_lwork, h_work, h_lwork, d_info);

        //check for errors
        if(solvererr == CUSOLVER_STATUS_SUCCESS){
            //std::cout << "success" << std::endl;
        }else if(solvererr == CUSOLVER_STATUS_INVALID_VALUE){
            std::cout << "invalid value" << std::endl;
        }
        int info = 0;
        cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    }

    void LinearSolverCuSolverDN::factorize(const Eigen::MatrixXd &A)
    {
        //find number of rows in A
        numrows = (int)A.rows();

        //copy A to device
        cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size());
        cudaMemcpy(d_A, (const void *)A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice);
        
        //allocate pivots
        cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * numrows);

        //calculate buffer size
        cusolverDnXgetrf_bufferSize(cuHandle, cuParams, numrows, numrows, CUDA_R_64F, d_A,
                                    numrows, CUDA_R_64F, &d_lwork, &h_lwork);
        cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * d_lwork);
        cudaMalloc(reinterpret_cast<void **>(&h_work), sizeof(double) * h_lwork);
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
        
        //factorize
        cusolverStatus_t solvererr = cusolverDnXgetrf(cuHandle, cuParams, numrows, numrows, CUDA_R_64F, d_A, 
        numrows, d_Ipiv, CUDA_R_64F, d_work, d_lwork, h_work, h_lwork, d_info);

        if(solvererr == CUSOLVER_STATUS_SUCCESS){
            //std::cout << "success" << std::endl;
        }else if(solvererr == CUSOLVER_STATUS_INVALID_VALUE){
            std::cout << "invalid value" << std::endl;
        }
        int info = 0;
        cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    }

    void LinearSolverCuSolverDN::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        //copy b to device
        cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * b.size());
        cudaMemcpy(d_b, (const void *)b.data(), sizeof(double) * b.size(), cudaMemcpyHostToDevice);

        //solve
        cusolverStatus_t solvererr = cusolverDnXgetrs(cuHandle, cuParams, CUBLAS_OP_N, numrows, 1,
            CUDA_R_64F, d_A, numrows, d_Ipiv,
            CUDA_R_64F, d_b, numrows, d_info);
        if(solvererr == CUSOLVER_STATUS_SUCCESS){
            //std::cout << "success" << std::endl;
        }else if(solvererr == CUSOLVER_STATUS_INVALID_VALUE){
            std::cout << "invalid value" << std::endl;
        }
        int info = 0;
        cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);

        //copy result to x
        cudaMemcpy(x.data(), d_b, sizeof(double) * x.size(), cudaMemcpyDeviceToHost);
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverCuSolverDN::~LinearSolverCuSolverDN()
    {
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_work);
        cudaFree(d_Ipiv);

        cusolverDnDestroyParams(cuParams);
        cusolverDnDestroy(cuHandle);
        cudaStreamDestroy(stream);
        cudaDeviceReset();
    }
    
}//namespace polysolve

#endif