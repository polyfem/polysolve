#include "RestartedAMGPCG.hpp"

#include <cuda_runtime.h>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <chrono>
#include <stdexcept>

#include <spdlog/spdlog.h>

// Internal header, needed for hypre_CTAlloc/hypre_TMemcpy so the dof_func
// array handed to HYPRE_BoomerAMGSetDofFunc uses the allocator HYPRE itself
// frees it with (and lives in the device memory HYPRE expects here).
#include "_hypre_utilities.h"

#ifdef HYPRE_ENABLE_MPI
#include <mpi.h>
#endif

#define CHECK_CUDA(call)                                                         \
    do                                                                           \
    {                                                                            \
        cudaError_t status = call;                                               \
        if (status != cudaSuccess)                                               \
        {                                                                        \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__         \
                      << " - " << cudaGetErrorName(status)                       \
                      << " (" << cudaGetErrorString(status) << ")" << std::endl; \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// Per-PCG-iteration logs (matmul, amg_v_cycle, pcg_iter) go through this
// instead of SPDLOG_INFO directly, so they can be silenced with
// detailed_log=false without affecting any other log line.
#define RESTARTED_AMG_PCG_LOG_ITER(...) \
    do                                  \
    {                                   \
        if (detailed_log)               \
            SPDLOG_INFO(__VA_ARGS__);   \
    } while (0)

namespace polysolve::linear
{

    namespace
    {
        using clock = std::chrono::steady_clock;

        double elapsed_seconds(const std::chrono::time_point<clock> &begin)
        {
            return std::chrono::duration<double>(clock::now() - begin).count();
        }

        // Error-free transformations: s + e == a + b and p + e == a * b
        // exactly (Knuth's TwoSum; TwoProduct via FMA). The explicit
        // __dadd_rn/__dsub_rn/__dmul_rn intrinsics are never contracted into
        // an FMA by nvcc, unlike plain +, -, *; such a contraction (e.g. of
        // a * b + c) would silently break the exactness these rely on.
        __device__ inline void two_sum(const double a, const double b, double &s, double &e)
        {
            s = __dadd_rn(a, b);
            const double bb = __dsub_rn(s, a);
            e = __dadd_rn(__dsub_rn(a, __dsub_rn(s, bb)), __dsub_rn(b, bb));
        }

        __device__ inline void two_prod(const double a, const double b, double &p, double &e)
        {
            p = __dmul_rn(a, b);
            e = __fma_rn(a, b, -p);
        }
    } // namespace

    RestartedAMGPCG::RestartedAMGPCG()
    {
#ifdef HYPRE_ENABLE_MPI
        int done_already;

        MPI_Initialized(&done_already);
        if (!done_already)
        {
            MPI_Init(nullptr, nullptr);
        }
#endif
        if (!HYPRE_Initialized())
        {
            HYPRE_Initialize();
        }

        HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
        HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
        HYPRE_SetSpGemmUseCusparse(false);
        HYPRE_SetUseGpuRand(true);
    }

    void RestartedAMGPCG::set_parameters(const json &params)
    {
        if (params.contains("RestartedAMGPCG"))
        {
            if (params["RestartedAMGPCG"].contains("max_iter"))
            {
                max_iter_ = params["RestartedAMGPCG"]["max_iter"];
            }
            if (params["RestartedAMGPCG"].contains("relative_tolerance"))
            {
                rel_conv_tol_ = params["RestartedAMGPCG"]["relative_tolerance"];
            }
            if (params["RestartedAMGPCG"].contains("absolute_tolerance"))
            {
                abs_conv_tol_ = params["RestartedAMGPCG"]["absolute_tolerance"];
            }
            if (params["RestartedAMGPCG"].contains("theta"))
            {
                theta = params["RestartedAMGPCG"]["theta"];
            }
            if (params["RestartedAMGPCG"].contains("restart_interval"))
            {
                const int interval = params["RestartedAMGPCG"]["restart_interval"];
                if (interval < 1)
                {
                    throw std::runtime_error("RestartedAMGPCG: restart_interval must be at least 1, got " + std::to_string(interval));
                }
                restart_interval = interval;
            }
            if (params["RestartedAMGPCG"].contains("detailed_log"))
            {
                detailed_log = params["RestartedAMGPCG"]["detailed_log"];
            }
        }
    }

    void RestartedAMGPCG::set_block_size(int block_size)
    {
        dimension_ = block_size;
    }

    void RestartedAMGPCG::set_block_mapping(const Eigen::VectorXi &block_mapping)
    {
        block_mapping_ = block_mapping;
    }

    void RestartedAMGPCG::get_info(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["num_restarts"] = num_restarts;
        params["final_res_norm"] = final_res_norm;
    }

    void RestartedAMGPCG::factorize(const StiffnessMatrix &Ain)
    {
        SPDLOG_INFO("[{}] [start_solve] [0.000000] [problem_size={}]", name(), Ain.rows());

        {
            auto phase_begin = clock::now();

            d_outer_indices.resize(Ain.rows() + 1);
            d_inner_indices.resize(Ain.nonZeros());
            d_values.resize(Ain.nonZeros());

            thrust::copy(Ain.outerIndexPtr(), Ain.outerIndexPtr() + d_outer_indices.size(), d_outer_indices.begin());
            thrust::copy(Ain.innerIndexPtr(), Ain.innerIndexPtr() + d_inner_indices.size(), d_inner_indices.begin());
            thrust::copy(Ain.valuePtr(), Ain.valuePtr() + d_values.size(), d_values.begin());

            SPDLOG_INFO("[{}] [copy_matrix_to_gpu] [{:.6f}]", name(), elapsed_seconds(phase_begin));
        }

        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
            A = nullptr;
        }

        copy_matrix_to_hypre();
        has_matrix_ = true;
    }

    namespace
    {
        // Same BoomerAMG configuration as GPUHybridSolver.
        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 8; // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
            int agg_levels = 1;   // number of aggressive coarsening levels
            double theta = 0.25;  // strength threshold: 0.25, 0.5, 0.8

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i, 0 = classical
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 18;  // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
            int relax_sweeps = 1; // relaxation sweeps on each level

            // Additional options:
            int print_level = 0; // print AMG iterations? 1 = no, 2 = yes
            int max_levels = 25; // max number of levels in AMG hierarchy

            int min_coarse_size = 5;

            HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
            HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);

            HYPRE_BoomerAMGSetRelaxOrder(amg_precond, false);
            HYPRE_BoomerAMGSetRAP2(amg_precond, true);
            HYPRE_BoomerAMGSetKeepTranspose(amg_precond, true);

            HYPRE_BoomerAMGSetMinCoarseSize(amg_precond, min_coarse_size);
            HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_type, 3);
            HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
            HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
            HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
            HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
            HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);

            // Use as a preconditioner (one V-cycle, zero tolerance)
            HYPRE_BoomerAMGSetMaxIter(amg_precond, 1);
            HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
        }

        void HypreBoomerAMG_SetElasticityOptions(HYPRE_Solver &amg_precond, int dim, double theta)
        {
            // Make sure the systems AMG options are set
            HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

            // More robust options with respect to convergence
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
        }
    } // namespace

    void RestartedAMGPCG::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        thrust::device_vector<double> d_x(x.size());
        thrust::device_vector<double> d_b(b.size());

        thrust::copy(x.data(), x.data() + x.size(), d_x.begin());
        thrust::copy(b.data(), b.data() + b.size(), d_b.begin());

        HYPRE_ParVector par_b;
        HYPRE_ParVector par_x;
        init_hypre_vectors(b.size());

        set_hypre_vec(ij_b, par_b, d_b);
        set_hypre_vec(ij_x, par_x, d_x);

        HYPRE_Solver precond;

        {
            auto phase_begin = clock::now();

            HYPRE_BoomerAMGCreate(&precond);
            HypreBoomerAMG_SetDefaultOptions(precond);
            if (dimension_ > 1)
            {
                HypreBoomerAMG_SetElasticityOptions(
                    precond,
                    dimension_,
                    theta);

                if (block_mapping_.size() > 0)
                {
                    assert(block_mapping_.size() == b.size());

                    // HYPRE_MEMORY_DEVICE is the active memory location here (see the
                    // constructor), so the array must live on the GPU. Build it on the
                    // host first, then copy it over with HYPRE's own allocator/copy so
                    // HYPRE can free it (via hypre_TFree) when the AMG precond is destroyed.
                    std::vector<HYPRE_Int> dof_func_host(b.size());
                    for (int i = 0; i < b.size(); ++i)
                        dof_func_host[i] = block_mapping_[i];

                    HYPRE_Int *dof_func = hypre_CTAlloc(HYPRE_Int, b.size(), HYPRE_MEMORY_DEVICE);
                    hypre_TMemcpy(dof_func, dof_func_host.data(), HYPRE_Int, b.size(),
                                  HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
                    HYPRE_BoomerAMGSetDofFunc(precond, dof_func);
                }
            }

            HYPRE_BoomerAMGSetup(precond, parcsr_A, par_b, par_x);
            CHECK_CUDA(cudaDeviceSynchronize());
            SPDLOG_INFO("[{}] [amg_setup] [{:.6f}]", name(), elapsed_seconds(phase_begin));
        }

        {
            auto phase_begin = clock::now();

            // Also sets final_res_norm, from the double-double residual it
            // last checked for convergence.
            pcg_solve(d_b, d_x, precond);

            thrust::copy(d_x.begin(), d_x.end(), x.data());

            CHECK_CUDA(cudaDeviceSynchronize());
            SPDLOG_INFO("[{}] [pcg_solve] [{:.6f}] [pcg_iters={}] [restarts={}] [residual={}]", name(), elapsed_seconds(phase_begin), num_iterations, num_restarts, final_res_norm);
        }

        {
            HYPRE_BoomerAMGDestroy(precond);
            HYPRE_IJVectorDestroy(ij_x);
            HYPRE_IJVectorDestroy(ij_b);
        }
    }

    void RestartedAMGPCG::copy_matrix_to_hypre()
    {
        auto phase_begin = clock::now();

        const HYPRE_Int num_rows = d_outer_indices.size() - 1;

#ifdef HYPRE_ENABLE_MPI
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, 0, num_rows - 1, 0, num_rows - 1, &A);
#else
        HYPRE_IJMatrixCreate(0, 0, num_rows - 1, 0, num_rows - 1, &A);
#endif
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        thrust::device_vector<HYPRE_Int> d_rows(num_rows);
        thrust::sequence(d_rows.begin(), d_rows.end());

        thrust::device_vector<HYPRE_Int> d_n_cols(num_rows);
        const HYPRE_Int *raw_outer = thrust::raw_pointer_cast(d_outer_indices.data());
        HYPRE_Int *raw_n_cols = thrust::raw_pointer_cast(d_n_cols.data());

        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(num_rows),
                         [=] __device__(int i) {
                             raw_n_cols[i] = raw_outer[i + 1] - raw_outer[i];
                         });

        HYPRE_Int *gpu_n_cols = thrust::raw_pointer_cast(d_n_cols.data());
        HYPRE_Int *gpu_rows = thrust::raw_pointer_cast(d_rows.data());

        HYPRE_Int *gpu_cols = thrust::raw_pointer_cast(d_inner_indices.data());
        double *gpu_vals = thrust::raw_pointer_cast(d_values.data());

        HYPRE_IJMatrixSetValues(A, num_rows, gpu_n_cols, gpu_rows, gpu_cols, gpu_vals);

        HYPRE_IJMatrixAssemble(A);

        void *temp_A = nullptr;
        HYPRE_IJMatrixGetObject(A, &temp_A);
        parcsr_A = static_cast<decltype(parcsr_A)>(temp_A);

        SPDLOG_INFO("[{}] [copy_matrix_to_hypre] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    void RestartedAMGPCG::init_hypre_vectors(const int size)
    {
#ifdef HYPRE_ENABLE_MPI
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, size - 1, &ij_x);
#else
        HYPRE_IJVectorCreate(0, 0, size - 1, &ij_x);
#endif
        HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
        HYPRE_IJVectorInitializeShell(ij_x);
#ifdef HYPRE_ENABLE_MPI
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, 0, size - 1, &ij_b);
#else
        HYPRE_IJVectorCreate(0, 0, size - 1, &ij_b);
#endif
        HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
        HYPRE_IJVectorInitializeShell(ij_b);
    }

    void RestartedAMGPCG::matmul(const thrust::device_vector<double> &x, thrust::device_vector<double> &result)
    {
        auto phase_begin = clock::now();
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_result;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_result, result);

        HYPRE_ParCSRMatrixMatvec(1.0, parcsr_A, par_x, 0.0, par_result);
        CHECK_CUDA(cudaDeviceSynchronize());
        RESTARTED_AMG_PCG_LOG_ITER("[{}] [matmul] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    double RestartedAMGPCG::dot(const thrust::device_vector<double> &a, const thrust::device_vector<double> &b)
    {
        HYPRE_ParVector par_a;
        HYPRE_ParVector par_b;

        set_hypre_vec(ij_x, par_a, a);
        set_hypre_vec(ij_b, par_b, b);

        double result;
        HYPRE_ParVectorInnerProd(par_a, par_b, &result);
        return result;
    }

    void RestartedAMGPCG::vector_copy(const thrust::device_vector<double> &x, thrust::device_vector<double> &y)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_y;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_y, y);

        HYPRE_ParVectorCopy(par_x, par_y);
    }

    void RestartedAMGPCG::vector_add(double alpha, const thrust::device_vector<double> &x, thrust::device_vector<double> &y)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_y;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_y, y);

        hypre_ParVectorAxpy(alpha, par_x, par_y);
    }

    void RestartedAMGPCG::vector_scale(double alpha, thrust::device_vector<double> &x)
    {
        HYPRE_ParVector par_x;

        set_hypre_vec(ij_x, par_x, x);

        HYPRE_ParVectorScale(alpha, par_x);
    }

    void RestartedAMGPCG::set_hypre_vec(HYPRE_IJVector &my_ij_x, HYPRE_ParVector &par_x, const thrust::device_vector<double> &x)
    {
        double *raw_ptr = const_cast<double *>(thrust::raw_pointer_cast(x.data()));

        HYPRE_IJVectorSetData(my_ij_x, raw_ptr);
        HYPRE_IJVectorAssemble(my_ij_x);
        HYPRE_IJVectorGetObject(my_ij_x, (void **)&par_x);
    }

    void RestartedAMGPCG::dd_residual(const thrust::device_vector<double> &b, const thrust::device_vector<double> &x, thrust::device_vector<double> &r)
    {
        const int num_rows = d_outer_indices.size() - 1;

        const int *raw_outer = thrust::raw_pointer_cast(d_outer_indices.data());
        const int *raw_inner = thrust::raw_pointer_cast(d_inner_indices.data());
        const double *raw_values = thrust::raw_pointer_cast(d_values.data());
        const double *raw_b = thrust::raw_pointer_cast(b.data());
        const double *raw_x = thrust::raw_pointer_cast(x.data());
        double *raw_r = thrust::raw_pointer_cast(r.data());

        // Ogita, Rump & Oishi's Dot2 ("Accurate sum and dot product", 2005),
        // seeded with b_i: the rounding error of every product and partial
        // sum is recovered exactly and accumulated into lo, so hi + lo is as
        // accurate as if b_i - (Ax)_i were computed in twice the working
        // precision and rounded once -- even when b_i and (Ax)_i nearly
        // cancel, which is exactly when a double-precision residual fails.
        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(num_rows),
                         [=] __device__(int i) {
                             double hi = raw_b[i];
                             double lo = 0.0;
                             for (int k = raw_outer[i]; k < raw_outer[i + 1]; ++k)
                             {
                                 double prod, prod_err, sum_err;
                                 two_prod(-raw_values[k], raw_x[raw_inner[k]], prod, prod_err);
                                 two_sum(hi, prod, hi, sum_err);
                                 lo += sum_err + prod_err;
                             }
                             raw_r[i] = hi + lo;
                         });
    }

    void RestartedAMGPCG::amg_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double> &b, thrust::device_vector<double> &x)
    {
        auto phase_begin = clock::now();
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_b;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_b, b);

        HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);
        CHECK_CUDA(cudaDeviceSynchronize());
        RESTARTED_AMG_PCG_LOG_ITER("[{}] [amg_v_cycle] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    void RestartedAMGPCG::pcg_solve(thrust::device_vector<double> &rhs, thrust::device_vector<double> &result, HYPRE_Solver &precond)
    {
        thrust::device_vector<double> r(rhs.size());
        thrust::device_vector<double> p(rhs.size());
        thrust::device_vector<double> z(rhs.size());
        thrust::device_vector<double> buffer(rhs.size());

        num_iterations = 0;
        num_restarts = 0;

        const double bi_prod = dot(rhs, rhs);
        if (bi_prod <= 0.0)
        {
            thrust::fill(result.begin(), result.end(), 0.0);
            final_res_norm = 0;
            return;
        }

        const double rel_eps = rel_conv_tol_ * rel_conv_tol_;
        const double abs_eps = abs_conv_tol_ * abs_conv_tol_;

        const auto converged_rel = [&](double r_prod) { return rel_eps > 0 && (r_prod / bi_prod) < rel_eps; };
        const auto converged_abs = [&](double r_prod) { return abs_eps > 0 && r_prod < abs_eps; };

        // Squared norm of the recursively-updated residual as of the last PCG
        // iteration, compared against the recomputed one at each restart.
        double i_prod = 0.0;
        bool breakdown = false;

        // Every pass through this loop's head (including the last, however
        // PCG stopped) recomputes the true residual in double-double, so
        // convergence is only declared on it, never on the recursive one.
        for (int cycle = 0;; ++cycle)
        {
            auto phase_begin = clock::now();

            dd_residual(rhs, result, r);
            const double r_prod = dot(r, r);
            final_res_norm = sqrt(r_prod);

            if (cycle == 0)
            {
                SPDLOG_INFO("[{}] [pre_loop] [{:.6f}] [rhs_norm={}] [residual={}]", name(), elapsed_seconds(phase_begin), sqrt(bi_prod), final_res_norm);
            }
            else
            {
                num_restarts = cycle;
                SPDLOG_INFO("[{}] [restart] [{:.6f}] [iter={}] [residual={}] [recursive_residual={}]", name(), elapsed_seconds(phase_begin), num_iterations, final_res_norm, sqrt(i_prod));
            }

            if (converged_rel(r_prod))
            {
                SPDLOG_INFO("[{}] [converged_rel] [0.000000]", name());
                break;
            }
            if (converged_abs(r_prod))
            {
                SPDLOG_INFO("[{}] [converged_abs] [0.000000]", name());
                break;
            }
            if (breakdown || num_iterations >= max_iter_)
            {
                break;
            }

            // Restart PCG from the recomputed residual: throw away the
            // previous search direction and start over from p = M^-1 r.
            thrust::fill(z.begin(), z.end(), 0.0);
            amg_precond_iter(precond, r, z);
            vector_copy(z, p);
            double gamma = dot(r, z);

            for (int k = 0; k < restart_interval && num_iterations < max_iter_; ++k)
            {
                auto iter_begin = clock::now();
                ++num_iterations;

                matmul(p, buffer);
                double sdotp = dot(p, buffer);

                if (sdotp == 0.0)
                {
                    SPDLOG_INFO("[{}] [err_zero_sdotp] [0.000000]", name());
                    breakdown = true;
                    break;
                }

                double alpha = gamma / sdotp;

                if (alpha <= 0.0)
                {
                    SPDLOG_INFO("[{}] [err_negative_alpha] [0.000000]", name());
                    breakdown = true;
                    break;
                }
                else if (alpha < __DBL_MIN__)
                {
                    SPDLOG_INFO("[{}] [err_subnormal_alpha] [0.000000]", name());
                    breakdown = true;
                    break;
                }

                vector_add(alpha, p, result);
                vector_add(-1.0 * alpha, buffer, r);
                i_prod = dot(r, r);

                // Only a hint: cut this cycle short and let the recomputed
                // residual at the loop head confirm (or refute) convergence.
                if (converged_rel(i_prod) || converged_abs(i_prod))
                {
                    break;
                }

                thrust::fill(z.begin(), z.end(), 0.0);
                amg_precond_iter(precond, r, z);

                double old_gamma = gamma;
                gamma = dot(r, z);

                double beta = gamma / old_gamma;

                vector_scale(beta, p);
                vector_add(1.0, z, p);

                CHECK_CUDA(cudaDeviceSynchronize());
                RESTARTED_AMG_PCG_LOG_ITER("[{}] [pcg_iter] [{:.6f}] [iter={}] [residual={}]", name(), elapsed_seconds(iter_begin), num_iterations - 1, sqrt(i_prod));
            }
        }
    }

    RestartedAMGPCG::~RestartedAMGPCG()
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
            A = nullptr;
        }
    }
} // namespace polysolve::linear
