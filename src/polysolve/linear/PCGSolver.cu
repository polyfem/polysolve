#include "PCGSolver.hpp"

// #ifndef SPDLOG_ACTIVE_LEVEL
// #define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
// #endif

#include <Eigen/Core>

#include <cub/cub.cuh>

#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/stream>
#include <cuda/memory_pool>
#include <cuda/std/cmath>
#include <cuda/std/span>
#include <cuda/std/optional>

#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include <polysolve/linear/mas_utils/BSRAdjacency.hpp>
#include <polysolve/linear/mas_utils/BSRMatrix.hpp>
#include <polysolve/linear/mas_utils/CuSparseWrapper.hpp>
#include <polysolve/linear/mas_utils/CudaUtils.cuh>
#include <polysolve/linear/mas_utils/InnerProduct.hpp>
#include <polysolve/linear/mas_utils/MASPreconditioner.hpp>
#include <polysolve/linear/mas_utils/MetisPartition.hpp>
#include <spdlog/spdlog.h>

namespace polysolve::linear
{

    using namespace mas;

    namespace
    {
        using clock = std::chrono::steady_clock;

        double elapsed_seconds(const std::chrono::time_point<clock> &begin)
        {
            return std::chrono::duration<double>(clock::now() - begin).count();
        }

        /// Device scalar devision num / denom.
        /// Defaults to zero when denom is small because we only check termination condition every 10
        /// PCG iterations and beta should be zero if we already arrive at solution.
        void scalar_division(
            ctd::span<const double> num,
            ctd::span<const double> denom,
            ctd::span<double> out,
            CudaRuntime rt)
        {
            auto op = [num, denom, out] __device__(int) {
                out[0] = (ctd::abs(denom[0]) < 1e-20) ? 0.0 : (num[0] / denom[0]);
            };
            cub::DeviceFor::Bulk(1, op, rt.stream.get());
        }

        /// Compute alpha * x + beta * y.
        /// @params h_alpha Host alpha.
        /// @params d_alpha Device alpha. nullptr implies 1.0.
        /// @params h_beta Host beta.
        /// @params d_beta Device beta. nullptr implies 1.0.
        /// @params x Device vector x.
        /// @params y Device vector y and the output.
        /// @params rt Cuda runtime.
        void axpby(
            double h_alpha,
            const double *d_alpha,
            double h_beta,
            const double *d_beta,
            ctd::span<const double> x,
            ctd::span<double> y,
            CudaRuntime rt)
        {
            auto op = [h_alpha, d_alpha, h_beta, d_beta, x, y] __device__(int idx) {
                double alpha = h_alpha * ((d_alpha == nullptr) ? 1.0 : *d_alpha);
                double beta = h_beta * ((d_beta == nullptr) ? 1.0 : *d_beta);
                y[idx] = alpha * x[idx] + beta * y[idx];
            };
            cub::DeviceFor::Bulk(x.size(), op, rt.stream.get());
        }

        /// @brief Build CSR row ptr style parition offset.
        /// Suppose all nodes are sorted by partition id, partition offset layout is:
        /// | partition 0 start | parition 1 start | ... |
        std::vector<int> build_part_offsets(ctd::span<const int> part_id, int part_num)
        {
            std::vector<int> offsets(part_num + 1, 0);
            for (int old_id = 0; old_id < part_id.size(); ++old_id)
            {
                offsets[part_id[old_id] + 1] += 1;
            }
            for (int part = 0; part < part_num; ++part)
            {
                offsets[part + 1] += offsets[part];
            }
            return offsets;
        }

        /// @brief Build permutation based such that all nodes are sorted by parition.
        std::vector<int> build_permutation(
            const std::vector<int> &part_id,
            const std::vector<int> &offsets)
        {
            std::vector<int> next = offsets;
            std::vector<int> permutation(part_id.size(), 0);
            for (int old_id = 0; old_id < part_id.size(); ++old_id)
            {
                int part = part_id[old_id];
                permutation[old_id] = next[part];
                next[part] += 1;
            }
            return permutation;
        }

        void permute_vector(
            ctd::span<const double> input,
            ctd::span<double> output,
            ctd::span<const int> block_map,
            int block_dim,
            CudaRuntime rt)
        {
            auto op = [input, output, block_map, block_dim] __device__(int idx) {
                int block_id = idx / block_dim;
                int comp = idx % block_dim;
                int mapped = block_map[block_id];
                output[idx] = input[mapped * block_dim + comp];
            };
            cub::DeviceFor::Bulk(output.size(), op, rt.stream.get());
        }

    } // namespace

    class CudaPCG::CudaPCGImpl
    {
    public:
        int block_dim_ = 3; ///< BSR block dim.
        int max_iter_ = 1e5;
        int true_residual_period_ = 4;
        double abs_tol_ = 1e-20;
        double rel_tol_ = 1e-6;
        bool lazy_partitioning_ = false;
        bool use_preconditioned_residual_norm_ = true;

        int dim_ = 0;          ///< Input matrix A dim.
        int permuted_dim_ = 0; ///< Dim with block padding.
        int iterations_ = 0;
        double residual_norm_ = 0.0;
        CudaPCGStatus status_ = CudaPCGStatus::Running;

        // == IMPORTANT ==
        // You must declare cu::stream and memory_pool before any device storage!
        // Else stream and mem pool will be destroyed before cuda::buffer, causing segfault.
        ctd::optional<cu::device_ref> default_device_;
        ctd::optional<cu::stream> default_stream_;
        ctd::optional<cu::device_memory_pool> default_mem_pool_;
        CuSparseHandle cusparse_handle_;

        BSRMatrix A_;
        MASPreconditioner mas_precond_;
        std::vector<int> permutation_;     ///< Permutation from metis partition.
        std::vector<int> inv_permutation_; ///< Permutation from metis partition.
        std::vector<int> part_offsets_;    ///< Metis partition offset for sorted blocks.
        CuSparseBSR sparse_A_;             ///< CuSparse handle of A.
        Buf<int> d_permutation_;           ///< Permutation from metis partition.
        Buf<int> d_inv_permutation_;       ///< Permutation from metis partition.
        Buf<char> spmv_workspace_;         ///< CuSparse temp.

        // PCG variables.

        Buf<double> x_;
        Buf<double> b_;
        Buf<double> r_;
        Buf<double> p_;
        Buf<double> z_;
        Buf<double> Ap_;
        Buf<char> reduction_storage_;
        Buf<double> scalar_rz_;
        Buf<double> scalar_pAp_;
        Buf<double> scalar_alpha_;
        Buf<double> scalar_beta_;
        Buf<double> scalar_rz_old_;
        Buf<double> scalar_rr_;

    public:
        CudaPCGImpl()
        {
            if (cu::devices.size() == 0)
            {
                throw std::runtime_error("No Nvidia GPU!!");
            }

            default_device_.emplace(cu::devices[0]);
            default_stream_.emplace(*default_device_);
            default_mem_pool_.emplace(*default_device_);
        }

        void set_parameters(const json &params)
        {
            if (params.contains("block_dim"))
                block_dim_ = params["block_dim"];
            if (params.contains("max_iter"))
                max_iter_ = params["max_iter"];
            if (params.contains("relative_tolerance"))
                rel_tol_ = params["relative_tolerance"];
            if (params.contains("absolute_tolerance"))
                abs_tol_ = params["absolute_tolerance"];
            if (params.contains("lazy_partitioning"))
                lazy_partitioning_ = params["lazy_partitioning"];
            if (params.contains("use_preconditioned_residual_norm"))
                use_preconditioned_residual_norm_ = params["use_preconditioned_residual_norm"];
        }

        void get_info(json &params) const
        {
            params["solver_iter"] = iterations_;
            params["solver_error"] = residual_norm_;
            params["solver_status"] = pcg_status_to_string(status_);
        }

        void analyze_pattern(const StiffnessMatrix &, const int) {}

        void build_partition_and_perm(const BSRAdjacency &adj)
        {
            int part_num = 0;
            std::vector<int> part_id;
            metis_partition(
                adj.row_ptr,
                adj.cols,
                adj.weights,
                32,
                part_num,
                part_id);

            part_offsets_ = build_part_offsets(part_id, part_num);
            permutation_ = build_permutation(part_id, part_offsets_);
            inv_permutation_.resize(permutation_.size());
            for (int old_id = 0; old_id < permutation_.size(); ++old_id)
            {
                inv_permutation_[permutation_[old_id]] = old_id;
            }
        }

        // Allocate cusparse workspace buffer.
        void setup_cusparse(CudaRuntime rt)
        {
            sparse_A_ = CuSparseBSR(A_.view());

            double alpha = 1.0;
            double beta = 0.0;
            CuSparseConstVec x_desc(*x_);
            CuSparseVec y_desc(*Ap_);
            size_t workspace_size = 0;

            cusparseSetStream(cusparse_handle_.raw, rt.stream.get());
            cusparseSpMV_bufferSize(
                cusparse_handle_.raw,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                sparse_A_.raw,
                x_desc.raw,
                &beta,
                y_desc.raw,
                CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                &workspace_size);

            spmv_workspace_ = cu::make_buffer<char>(
                rt.stream,
                rt.mr,
                workspace_size == 0 ? 1 : workspace_size,
                cu::no_init);
        }

        void spmv(ctd::span<const double> x, ctd::span<double> y, CudaRuntime rt)
        {
            double alpha = 1.0;
            double beta = 0.0;
            CuSparseConstVec x_desc(x);
            CuSparseVec y_desc(y);

            cusparseSetStream(cusparse_handle_.raw, rt.stream.get());
            cusparseSpMV(
                cusparse_handle_.raw,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha,
                sparse_A_.raw,
                x_desc.raw,
                &beta,
                y_desc.raw,
                CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                spmv_workspace_->data());
        }

        void factorize(const StiffnessMatrix &A)
        {
            CudaRuntime rt{*default_stream_, default_mem_pool_->as_ref()};
            auto total_begin = clock::now();
            int block_n = div_round_up(A.rows(), block_dim_);

            // We do:
            // 1. K-way graph parition.
            // 2. Build permutation based on partition.
            // 3. Initialize MAS preconditioner.
            // 4. Allocates buffer for PCG loop.

            bool rebuild_partition =
                !lazy_partitioning_ || permutation_.size() != block_n || part_offsets_.empty();
            if (rebuild_partition)
            {
                // Build adjacency for metis parition.
                auto phase_begin = clock::now();
                BSRAdjacency adj{A, block_dim_};
                SPDLOG_TRACE("CUDA_PCG setup: topology_adj {:.6f}s", elapsed_seconds(phase_begin));

                // Sort nodes based on parition.
                phase_begin = clock::now();
                build_partition_and_perm(adj);
                SPDLOG_INFO("CUDA_PCG setup: metis_partition {:.6f}s", elapsed_seconds(phase_begin));
            }
            else
            {
                SPDLOG_TRACE("CUDA_PCG setup: reuse_partition");
            }

            // Build new sorted BSR matrix for MAS initialization.
            auto phase_begin = clock::now();
            A_ = BSRMatrix{
                A,
                block_dim_,
                ctd::span<const int>(permutation_.data(), permutation_.size()),
                rt};
            SPDLOG_TRACE("CUDA_PCG setup: permuted_bsr {:.6f}s", elapsed_seconds(phase_begin));

            BSRView view = A_.view();
            dim_ = A.rows();
            permuted_dim_ = view.block_dim * view.dim;

            // Initialize MAS.
            phase_begin = clock::now();
            mas_precond_.factorize(
                A_,
                ctd::span<const int>(part_offsets_.data(), part_offsets_.size()),
                rt);
            rt.stream.sync();
            SPDLOG_TRACE("CUDA_PCG setup: mas_factorize {:.6f}s", elapsed_seconds(phase_begin));

            // Copy permutation to device.
            phase_begin = clock::now();
            d_permutation_ = cu::make_buffer<int>(rt.stream, rt.mr, permutation_.size(), cu::no_init);
            d_inv_permutation_ =
                cu::make_buffer<int>(rt.stream, rt.mr, inv_permutation_.size(), cu::no_init);
            cu::copy_bytes(rt.stream, permutation_, *d_permutation_);
            cu::copy_bytes(rt.stream, inv_permutation_, *d_inv_permutation_);

            // Allocates buffers for PCG loop.
            x_ = cu::make_buffer<double>(rt.stream, rt.mr, permuted_dim_, cu::no_init);
            b_ = cu::make_buffer<double>(rt.stream, rt.mr, permuted_dim_, cu::no_init);
            r_ = cu::make_buffer<double>(rt.stream, rt.mr, permuted_dim_, cu::no_init);
            p_ = cu::make_buffer<double>(rt.stream, rt.mr, permuted_dim_, cu::no_init);
            z_ = cu::make_buffer<double>(rt.stream, rt.mr, permuted_dim_, cu::no_init);
            Ap_ = cu::make_buffer<double>(rt.stream, rt.mr, permuted_dim_, cu::no_init);

            scalar_rz_ = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);
            scalar_pAp_ = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);
            scalar_alpha_ = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);
            scalar_beta_ = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);
            scalar_rz_old_ = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);
            scalar_rr_ = cu::make_buffer<double>(rt.stream, rt.mr, 1, cu::no_init);
            rt.stream.sync();
            SPDLOG_TRACE("CUDA_PCG setup: device_buffers {:.6f}s", elapsed_seconds(phase_begin));

            // Allocates buffers for CuSparse.
            phase_begin = clock::now();
            setup_cusparse(rt);
            rt.stream.sync();
            SPDLOG_TRACE("CUDA_PCG setup: cusparse {:.6f}s", elapsed_seconds(phase_begin));
            SPDLOG_TRACE("CUDA_PCG setup: total {:.6f}s", elapsed_seconds(total_begin));
        }

        void solve(const Eigen::Ref<const Eigen::VectorXd> b, Eigen::Ref<Eigen::VectorXd> x)
        {
            status_ = CudaPCGStatus::Running;

            if (b.size() != x.size() || !check_buffer_size(b.size()))
            {
                throw std::runtime_error("[CudaPCG] Size mismatch. Did you forget to call factorize?");
            }

            CudaRuntime rt{*default_stream_, default_mem_pool_->as_ref()};

            cu::copy_bytes(
                rt.stream,
                ctd::span<const double>(b.data(), dim_),
                ctd::span<double>(r_->data(), dim_));
            cu::fill_bytes(
                rt.stream, ctd::span<double>(r_->data() + dim_, permuted_dim_ - dim_), 0);
            permute_vector(*r_, *b_, *d_inv_permutation_, A_.view().block_dim, rt);

            // The solver sometimes fails to converge if we use input x as initial value.
            // Maybe the caller does not initialize x properly?
            // Set initial x to zero to work around this issue for now.
            cu::fill_bytes(rt.stream, *x_, 0);

            pcg_solve(rt);

            permute_vector(*x_, *r_, *d_permutation_, A_.view().block_dim, rt);

            cu::copy_bytes(
                rt.stream,
                ctd::span<const double>(r_->data(), dim_),
                ctd::span<double>(x.data(), dim_));
            rt.stream.sync();
        }

        bool check_buffer_size(int n) const
        {
            if (n <= 0 || mas_precond_.empty() || !x_ || !b_ || !r_ || !p_ || !z_ || !Ap_
                || !scalar_rz_ || !scalar_pAp_ || !scalar_alpha_ || !scalar_beta_
                || !scalar_rz_old_ || !scalar_rr_)
            {
                return false;
            }

            BSRView view = A_.view();
            int block_n = (n + view.block_dim - 1) / view.block_dim;
            int block_size = view.block_dim * view.block_dim;

            if (n != dim_ || block_n != view.dim)
            {
                return false;
            }

            if (view.rows.size() != (view.dim + 1)
                || view.cols.size() != view.non_zeros
                || view.vals.size() != block_size * view.non_zeros)
            {
                return false;
            }

            if (!d_permutation_ || !d_inv_permutation_ || sparse_A_.raw == nullptr || !spmv_workspace_
                || x_->size() != permuted_dim_
                || b_->size() != permuted_dim_
                || r_->size() != permuted_dim_
                || p_->size() != permuted_dim_
                || z_->size() != permuted_dim_
                || Ap_->size() != permuted_dim_)
            {
                return false;
            }

            if (scalar_rz_->size() < 1
                || scalar_pAp_->size() < 1
                || scalar_alpha_->size() < 1
                || scalar_beta_->size() < 1
                || scalar_rz_old_->size() < 1
                || scalar_rr_->size() < 1)
            {
                return false;
            }

            return true;
        }

        /// https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
        void pcg_solve(CudaRuntime rt)
        {
            // Compute initial residual r = b-Ax.
            spmv(*x_, *r_, rt);
            axpby(1.0, nullptr, -1.0, nullptr, *b_, *r_, rt);

            // Compute z = M^-1 r.
            mas_precond_.apply(*r_, *z_, rt);
            // Initial search direction p = z;
            cu::copy_bytes(rt.stream, *z_, *p_);

            // Compute rz = r^T M^-1 r.
            inner_product(*r_, *z_, *scalar_rz_, rt);
            const double rz0 = device2host(scalar_rz_->data(), rt);
            if (ctd::isnan(rz0) || !ctd::isfinite(rz0))
            {
                throw std::runtime_error("[CudaPCG] Invalid initial residual.");
            }

            // Compute rr = r^T r.
            double rr0 = 0.0;
            if (!use_preconditioned_residual_norm_)
            {
                inner_product(*r_, *r_, *scalar_rr_, rt);
                rr0 = device2host(scalar_rr_->data(), rt);
            }

            // debug logging
            auto iter_window_begin = clock::now();
            int iter_window_start = 1;

            for (int k = 1; k <= max_iter_; ++k)
            {
                // Compute Ap = A p.
                spmv(*p_, *Ap_, rt);
                // Compute pAp = p^T * A * p.
                inner_product(*p_, *Ap_, *scalar_pAp_, rt);
                // Compute alpha = (r M^-1 r) / (p^T A p).
                scalar_division(*scalar_rz_, *scalar_pAp_, *scalar_alpha_, rt);
                // Compute x = x + alpha A p.
                axpby(1.0, scalar_alpha_->data(), 1.0, nullptr, *p_, *x_, rt);

                // Compute residual b-Ax directly.
                if (k % true_residual_period_ == 0)
                {
                    spmv(*x_, *r_, rt);
                    axpby(1.0, nullptr, -1.0, nullptr, *b_, *r_, rt);
                }
                // Compute residual update using r' = r - alpha A p.
                // This saves one spmv but accumulates floating point error overtime.
                else
                {
                    axpby(-1.0, scalar_alpha_->data(), 1.0, nullptr, *Ap_, *r_, rt);
                }

                // Compute z = M^-1 r.
                mas_precond_.apply(*r_, *z_, rt);

                // Compute rz = r M^-1 r.
                cu::copy_bytes(rt.stream, *scalar_rz_, *scalar_rz_old_);
                inner_product(*r_, *z_, *scalar_rz_, rt);

                iterations_ = k;
                bool converged = false;

                // Check convergence every 10 iterations.
                if (k % 10 == 0)
                {
                    if (use_preconditioned_residual_norm_)
                    {
                        double rz_new = device2host(scalar_rz_->data(), rt);
                        residual_norm_ = ctd::sqrt(rz_new);
                        if (rz_new <= rel_tol_ * rel_tol_ * rz0 || rz_new <= abs_tol_ * abs_tol_)
                        {
                            status_ = (rz_new <= abs_tol_ * abs_tol_)
                                          ? CudaPCGStatus::ReachAbsoluteTolerance
                                          : CudaPCGStatus::ReachRelativeTolerance;
                            converged = true;
                        }
                    }
                    else
                    {
                        inner_product(*r_, *r_, *scalar_rr_, rt);
                        double rr = device2host(scalar_rr_->data(), rt);
                        residual_norm_ = ctd::sqrt(rr);
                        if (rr <= rel_tol_ * rel_tol_ * rr0 || rr <= abs_tol_ * abs_tol_)
                        {
                            status_ = (rr <= abs_tol_ * abs_tol_)
                                          ? CudaPCGStatus::ReachAbsoluteTolerance
                                          : CudaPCGStatus::ReachRelativeTolerance;
                            converged = true;
                        }
                    }
                }

                // debug logging
                if (k % 100 == 0)
                {
                    rt.stream.sync();
                    SPDLOG_TRACE(
                        "CUDA_PCG iter {}-{}: {:.6f}s residual {:.6e}",
                        iter_window_start,
                        k,
                        elapsed_seconds(iter_window_begin),
                        residual_norm_);
                    iter_window_begin = clock::now();
                    iter_window_start = k + 1;
                }

                if (converged)
                {
                    break;
                }

                // Compute beta = rz / rz_old.
                scalar_division(*scalar_rz_, *scalar_rz_old_, *scalar_beta_, rt);
                // Compute direction update p' = M^-1 r + beta p.
                axpby(1.0, nullptr, 1.0, scalar_beta_->data(), *z_, *p_, rt);
            }

            if (iterations_ == max_iter_)
            {
                status_ = CudaPCGStatus::ReachMaxIterations;
            }

            SPDLOG_TRACE(
                "CUDA_PCG iter {}-{}: {:.6f}s residual {:.6e}",
                iter_window_start,
                iterations_,
                elapsed_seconds(iter_window_begin),
                residual_norm_);
        }
    };

    CudaPCG::CudaPCG()
        : impl_(std::make_unique<CudaPCGImpl>())
    {
    }

    CudaPCG::~CudaPCG() = default;

    void CudaPCG::set_parameters(const json &params)
    {
        const std::string solver_name = name();
        if (!params.contains(solver_name))
        {
            return;
        }

        impl_->set_parameters(params[solver_name]);
    }

    void CudaPCG::get_info(json &params) const
    {
        impl_->get_info(params);
    }

    void CudaPCG::analyze_pattern(const StiffnessMatrix &A, const int precond_num)
    {
        impl_->analyze_pattern(A, precond_num);
    }

    void CudaPCG::factorize(const StiffnessMatrix &A)
    {
        impl_->factorize(A);
    }

    void CudaPCG::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        impl_->solve(b, x);
    }

    std::string CudaPCG::name() const
    {
        return "CUDA_PCG";
    }

} // namespace polysolve::linear
