#ifdef POLYSOLVE_WITH_CUSOLVER

////////////////////////////////////////////////////////////////////////////////
#include "LinearSolverCuSolverDN.cuh"

#include <Eigen/Dense>
#include <Eigen/Core>

#include <fstream>
#include <string>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{

    namespace
    {
        inline void gpuErrchk(cudaError_t code)
        {
            if (code != cudaSuccess)
            {
                throw cudaGetErrorString(code);
            }
        }

        template <typename T>
        inline cudaDataType_t cuda_type()
        {
            throw std::runtime_error("This should never be called!");
            return CUDA_R_64F;
        }

        template <>
        inline cudaDataType_t cuda_type<float>()
        {
            return CUDA_R_32F;
        }

        template <>
        inline cudaDataType_t cuda_type<double>()
        {
            return CUDA_R_64F;
        }

        template <typename T>
        inline Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> convert(const Eigen::MatrixXd &A)
        {
            return A.cast<T>();
        }

        template <>
        inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> convert(const Eigen::MatrixXd &A)
        {
            return A;
        }

        template <typename T>
        inline const Eigen::Matrix<T, Eigen::Dynamic, 1> convert_vec(const Eigen::Ref<const Eigen::VectorXd> b)
        {
            return b.cast<T>();
        }

        template <>
        inline const Eigen::Matrix<double, Eigen::Dynamic, 1> convert_vec(const Eigen::Ref<const Eigen::VectorXd> b)
        {
            return b;
        }

        template <typename T>
        inline const Eigen::VectorXd convert_back(const Eigen::Matrix<T, Eigen::Dynamic, 1> &b)
        {
            const Eigen::Matrix<double, Eigen::Dynamic, 1> tmp; // = b.cast<>();
            throw std::runtime_error("This should never be called!");
            return tmp;
        }

        template <>
        inline const Eigen::VectorXd convert_back(const Eigen::Matrix<double, Eigen::Dynamic, 1> &b)
        {
            return b;
        }

        template <>
        inline const Eigen::VectorXd convert_back(const Eigen::Matrix<float, Eigen::Dynamic, 1> &b)
        {
            return b.cast<double>();
        }

    } // namespace

    template <typename T>
    LinearSolverCuSolverDN<T>::LinearSolverCuSolverDN()
    {
        init();
    }

    template <typename T>
    void LinearSolverCuSolverDN<T>::init()
    {
        cusolverDnCreate(&cuHandle);
        cusolverDnCreateParams(&cuParams);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cusolverDnSetStream(cuHandle, stream);
    }

    template <typename T>
    void LinearSolverCuSolverDN<T>::getInfo(json &params) const
    {
    }

    template <typename T>
    void LinearSolverCuSolverDN<T>::factorize(const StiffnessMatrix &A)
    {
        factorize_dense(Eigen::MatrixXd(A));
    }

    template <typename T>
    void LinearSolverCuSolverDN<T>::factorize_dense(const Eigen::MatrixXd &A)
    {
        numrows = (int)A.rows();

        // copy A to device
        if (!d_A_alloc)
        {
            gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(T) * A.size()));
        }
        gpuErrchk(cudaMemcpy(d_A, (const void *)convert<T>(A).data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice));

        cusolverDnXgetrf_bufferSize(cuHandle, cuParams, numrows, numrows, cuda_type<T>(), d_A, numrows, cuda_type<T>(), &d_lwork, &h_lwork);
        if (!d_A_alloc)
        {
            gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(T) * d_lwork));
            gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&h_work), sizeof(T) * h_lwork));
            gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
            gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * numrows));
        }
        int info = 0;

        // factorize
        cusolverStatus_t solvererr = cusolverDnXgetrf(cuHandle, cuParams, numrows, numrows, cuda_type<T>(), d_A,
                                                      numrows, d_Ipiv, cuda_type<T>(), d_work, d_lwork, h_work, h_lwork, d_info);

        if (solvererr == CUSOLVER_STATUS_INVALID_VALUE)
        {
            throw std::invalid_argument("CUDA returned invalid value");
        }

        gpuErrchk(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

        d_A_alloc = true;
    }

    template <typename T>
    void LinearSolverCuSolverDN<T>::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        // copy b to device
        if (!d_b_alloc)
            gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(T) * b.size()));
        gpuErrchk(cudaMemcpy(d_b, (const void *)convert_vec<T>(b).data(), sizeof(T) * b.size(), cudaMemcpyHostToDevice));

        // solve
        cusolverStatus_t solvererr = cusolverDnXgetrs(cuHandle, cuParams, CUBLAS_OP_N, numrows, 1,
                                                      cuda_type<T>(), d_A, numrows, d_Ipiv,
                                                      cuda_type<T>(), d_b, numrows, d_info);
        if (solvererr == CUSOLVER_STATUS_INVALID_VALUE)
        {
            throw std::invalid_argument("CUDA returned invalid value");
        }
        int info = 0;
        gpuErrchk(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

        // copy result to x
        Eigen::Matrix<T, Eigen::Dynamic, 1> x_tmp(x.size());
        gpuErrchk(cudaMemcpy(x_tmp.data(), d_b, sizeof(T) * x.size(), cudaMemcpyDeviceToHost));
        x = convert_back<T>(x_tmp);

        d_b_alloc = true;
    }

    ////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    LinearSolverCuSolverDN<T>::~LinearSolverCuSolverDN()
    {
        if (d_A_alloc)
        {
            gpuErrchk(cudaFree(d_A));

            gpuErrchk(cudaFree(d_work));
            gpuErrchk(cudaFree(h_work));
            gpuErrchk(cudaFree(d_Ipiv));
            gpuErrchk(cudaFree(d_info));
        }

        cusolverDnDestroyParams(cuParams);
        cusolverDnDestroy(cuHandle);
        cudaStreamDestroy(stream);

        if (d_b_alloc)
            gpuErrchk(cudaFree(d_b));

        cudaDeviceReset();
    }

} // namespace polysolve

template polysolve::LinearSolverCuSolverDN<double>::LinearSolverCuSolverDN();
template polysolve::LinearSolverCuSolverDN<float>::LinearSolverCuSolverDN();

#endif