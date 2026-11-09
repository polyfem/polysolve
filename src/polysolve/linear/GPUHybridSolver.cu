

#include "GPUHybridSolver.hpp"

#include "hybrid_utils/DisjointSet.hpp"

#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/inner_product.h>
#include <thrust/distance.h>
#include <thrust/binary_search.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/functional.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>

#include <chrono>
#include <stdexcept>

#include <spdlog/spdlog.h>

// Internal header, needed for hypre_CTAlloc/hypre_TFree/hypre_TMemcpy so the
// dof_func array handed to HYPRE_BoomerAMGSetDofFunc uses the allocator HYPRE
// itself frees it with (and lives in the device memory HYPRE expects here).
#include "_hypre_utilities.h"
#include "_hypre_parcsr_ls.h" // hypre_ParAMGData, for the AMG hierarchy statistics

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

#define CHECK_CUDSS(call)                                                 \
    do                                                                    \
    {                                                                     \
        cudssStatus_t status = call;                                      \
        if (status != CUDSS_STATUS_SUCCESS)                               \
        {                                                                 \
            std::cerr << "cuDSS Error at " << __FILE__ << ":" << __LINE__ \
                      << " code " << (int)status << std::endl;            \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define CHECK_CUBLAS(call)                                                 \
    do                                                                     \
    {                                                                      \
        cublasStatus_t status = call;                                     \
        if (status != CUBLAS_STATUS_SUCCESS)                              \
        {                                                                 \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ \
                      << " code " << (int)status << std::endl;            \
            exit(EXIT_FAILURE);                                          \
        }                                                                 \
    } while (0)

#define CHECK_CUSPARSE(call)                                                 \
    do                                                                        \
    {                                                                         \
        cusparseStatus_t status = call;                                       \
        if (status != CUSPARSE_STATUS_SUCCESS)                                \
        {                                                                     \
            std::cerr << "cuSPARSE Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cusparseGetErrorString(status) << std::endl; \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

    // Per-PCG-iteration timing logs (matmul, amg_v_cycle, subdomain_solve,
    // contact_patch_dense_subdomain_solve, pcg_iter) go through this instead
    // of SPDLOG_INFO directly, so they can be silenced with
    // detailed_log=false without affecting any other log line.
#define GPUHYBRID_LOG_ITER(...)     \
    do                              \
    {                               \
        if (detailed_log)           \
            SPDLOG_INFO(__VA_ARGS__); \
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
    } // namespace

    GPUHybridSolver::GPUHybridSolver()
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

        CHECK_CUDSS(cudssCreate(&cudss_handle));
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
        CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
    }

    void GPUHybridSolver::set_parameters(const json &params)
    {
        if (params.contains("GPUHybrid"))
        {
            if (params["GPUHybrid"].contains("max_iter"))
            {
                max_iter_ = params["GPUHybrid"]["max_iter"];
            }
            if (params["GPUHybrid"].contains("relative_tolerance"))
            {
                rel_conv_tol_ = params["GPUHybrid"]["relative_tolerance"];
            }
            if (params["GPUHybrid"].contains("absolute_tolerance"))
            {
                abs_conv_tol_ = params["GPUHybrid"]["absolute_tolerance"];
            }
            if (params["GPUHybrid"].contains("theta"))
            {
                theta = params["GPUHybrid"]["theta"];
            }
            if (params["GPUHybrid"].contains("decompose_subdomains"))
            {
                decompose_subdomains = params["GPUHybrid"]["decompose_subdomains"];
            }
            if (params["GPUHybrid"].contains("min_subdomain_size"))
            {
                min_subdomain_size = params["GPUHybrid"]["min_subdomain_size"];
            }
            if (params["GPUHybrid"].contains("max_subdomain_size"))
            {
                max_subdomain_size = params["GPUHybrid"]["max_subdomain_size"];
            }
            if (params["GPUHybrid"].contains("gmm_jump_threshold"))
            {
                gmm_jump_threshold = params["GPUHybrid"]["gmm_jump_threshold"];
            }
            if (params["GPUHybrid"].contains("expand_subdomains"))
            {
                expand_subdomains = params["GPUHybrid"]["expand_subdomains"];
            }
            if (params["GPUHybrid"].contains("gmm_tol"))
            {
                gmm_tol = params["GPUHybrid"]["gmm_tol"];
            }
            if (params["GPUHybrid"].contains("max_gmm_iterations"))
            {
                max_gmm_iterations = params["GPUHybrid"]["max_gmm_iterations"];
            }
            if (params["GPUHybrid"].contains("conditioning_threshold"))
            {
                conditioning_threshold = params["GPUHybrid"]["conditioning_threshold"];
            }
            if (params["GPUHybrid"].contains("additive_mode"))
            {
                additive_mode = params["GPUHybrid"]["additive_mode"];
            }
            if (params["GPUHybrid"].contains("contact_patch_schwarz"))
            {
                contact_patch_schwarz = params["GPUHybrid"]["contact_patch_schwarz"];
            }
            if (params["GPUHybrid"].contains("detailed_log"))
            {
                detailed_log = params["GPUHybrid"]["detailed_log"];
            }
            if (params["GPUHybrid"].contains("krylov"))
            {
                const std::string str = params["GPUHybrid"]["krylov"];
                if (str == "pcg")
                    krylov_method = KrylovMethod::PCG;
                else if (str == "gmres")
                    krylov_method = KrylovMethod::GMRES;
                else
                    throw std::runtime_error("Unknown krylov method: " + str);
            }
            if (params["GPUHybrid"].contains("preconditioner"))
            {
                const std::string str = params["GPUHybrid"]["preconditioner"];
                if (str == "hybrid")
                    preconditioner_type = PreconditionerType::HYBRID;
                else if (str == "jacobi")
                    preconditioner_type = PreconditionerType::JACOBI;
                else if (str == "ilu")
                    preconditioner_type = PreconditionerType::ILU;
                else if (str == "ic")
                    preconditioner_type = PreconditionerType::IC;
                else
                    throw std::runtime_error("Unknown preconditioner: " + str);
            }
            if (params["GPUHybrid"].contains("direct"))
            {
                const std::string str = params["GPUHybrid"]["direct"];
                if (str == "none")
                    direct_factorization = DirectFactorization::NONE;
                else if (str == "lu")
                    direct_factorization = DirectFactorization::LU;
                else if (str == "llt")
                    direct_factorization = DirectFactorization::LLT;
                else
                    throw std::runtime_error("Unknown direct factorization: " + str);
            }
            if (params["GPUHybrid"].contains("gmres_restart"))
            {
                gmres_restart = params["GPUHybrid"]["gmres_restart"];
            }
            if (params["GPUHybrid"].contains("ilu_level_of_fill"))
            {
                ilu_level_of_fill = params["GPUHybrid"]["ilu_level_of_fill"];
            }
            if (params["GPUHybrid"].contains("ic_initial_shift"))
            {
                ic_initial_shift = params["GPUHybrid"]["ic_initial_shift"];
            }
            if (params["GPUHybrid"].contains("record_residual_history"))
            {
                record_residual_history = params["GPUHybrid"]["record_residual_history"];
            }
            if (params["GPUHybrid"].contains("select_bad_dofs_from_l1_norm"))
            {
                select_bad_dofs_from_l1_row_norm = params["GPUHybrid"]["select_bad_dofs_from_l1_norm"];
            }
            if (params["GPUHybrid"].contains("subdomain_selection_strategy"))
            {
                const std::string strategy_str = params["GPUHybrid"]["subdomain_selection_strategy"];
                if (strategy_str == "knee")
                {
                    subdomain_selection_strategy = SubdomainSelectionStrategy::KNEE;
                }
                else if (strategy_str == "gmm")
                {
                    subdomain_selection_strategy = SubdomainSelectionStrategy::GMM;
                }
                else if (strategy_str == "fd")
                {
                    subdomain_selection_strategy = SubdomainSelectionStrategy::FD;
                }
                else if (strategy_str == "cost")
                {
                    subdomain_selection_strategy = SubdomainSelectionStrategy::COST;
                }
                else if (strategy_str == "aposteriori")
                {
                    subdomain_selection_strategy = SubdomainSelectionStrategy::APOSTERIORI;
                }
                else
                {
                    throw std::runtime_error("Unknown subdomain selection strategy: " + strategy_str);
                }
            }
        }
    }

    void GPUHybridSolver::set_block_size(int block_size)
    {
        dimension_ = block_size;
    }

    void GPUHybridSolver::set_block_mapping(const Eigen::VectorXi &block_mapping)
    {
        block_mapping_ = block_mapping;
    }

    void GPUHybridSolver::get_info(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
        params["stop_reason"] = stop_reason;
        for (auto it = info_.begin(); it != info_.end(); ++it)
        {
            params[it.key()] = it.value();
        }
        if (record_residual_history)
        {
            params["residual_history"] = residual_history;
            params["time_history"] = time_history;
        }
    }

    void GPUHybridSolver::check_settings() const
    {
        if (subdomain_selection_strategy == SubdomainSelectionStrategy::APOSTERIORI)
        {
            throw std::runtime_error("A Posteriori subdomain selection strategy is not yet implemented!");
        }
        if (gmres_restart < 1)
        {
            throw std::runtime_error("gmres_restart must be at least 1!");
        }
    }

    void GPUHybridSolver::factorize(const StiffnessMatrix &Ain)
    {
        check_settings();
        info_ = json::object();
        SPDLOG_INFO("[{}] [start_solve] [0.000000] [problem_size={}]", name(), Ain.rows());

        // At = A^T in CSC is A in CSR: the row pointers, column indices and
        // values that hypre, cuDSS and the row-wise kernels below expect, and
        // its "column k" is row k of A for the host-side subdomain code.
        // (Ain's own arrays read as CSR would be A^T -- wrong for
        // nonsymmetric matrices.)
        const StiffnessMatrix At = Ain.transpose();

        if (direct_factorization != DirectFactorization::NONE)
        {
            factorize_direct(At);
            return;
        }

        {
            auto phase_begin = clock::now();

            d_outer_indices.resize(At.rows() + 1);
            d_inner_indices.resize(At.nonZeros());
            d_values.resize(At.nonZeros());

            thrust::copy(At.outerIndexPtr(), At.outerIndexPtr() + d_outer_indices.size(), d_outer_indices.begin());
            thrust::copy(At.innerIndexPtr(), At.innerIndexPtr() + d_inner_indices.size(), d_inner_indices.begin());
            thrust::copy(At.valuePtr(), At.valuePtr() + d_values.size(), d_values.begin());

            SPDLOG_INFO("[{}] [copy_matrix_to_gpu] [{:.6f}]", name(), elapsed_seconds(phase_begin));
        }

        if (preconditioner_type != PreconditionerType::HYBRID)
        {
            // No subdomains: custom_mixed_precond_iter is never called.
            auto phase_begin = clock::now();
            free_device_memory();
            free_contact_patch_dense_memory();
            bad_indices_arrays.clear();
            d_all_bad_dofs.clear();
            h_subdomain_sizes.clear();
            d_subdomain_sizes.clear();
            sparse_batch_count = 0;

            if (preconditioner_type == PreconditionerType::JACOBI)
                setup_jacobi(At);
            else if (preconditioner_type == PreconditionerType::IC)
                setup_ic(At);
            info_["subdomain_or_local_precond_setup_time"] = elapsed_seconds(phase_begin);
        }
        else
        {
            auto phase_begin = clock::now();

            bad_indices_arrays.clear();
            select_bad_dofs();

            if (contact_patch_schwarz)
            {
                for (const auto &patch : contact_patches)
                {
                    bad_indices_arrays.emplace_back(patch.begin(), patch.end());
                }
            }

            if (decompose_subdomains)
            {
                filter_subdomains(At);
            }

            if (expand_subdomains)
            {
                expand_subdomains_to_strongly_connected(At);
            }

            if (decompose_subdomains)
            {
                decompose_subdomains_to_disjoint_subsets(At);
            }
            else if (!contact_patch_schwarz)
            {
                bad_indices_arrays.emplace_back(h_all_bad_dofs.begin(), h_all_bad_dofs.end());
            }

            if (contact_patch_schwarz)
            {
                // Reads true columns (column-major dense blocks), so it takes Ain.
                factorize_contact_patches_dense(Ain);
            }
            else
            {
                d_all_bad_dofs.clear();
                h_subdomain_sizes.clear();
                d_subdomain_sizes.clear();

                for (int i = 0; i < bad_indices_arrays.size(); ++i)
                {
                    d_all_bad_dofs.insert(d_all_bad_dofs.end(), bad_indices_arrays[i].begin(), bad_indices_arrays[i].end());
                    h_subdomain_sizes.push_back(bad_indices_arrays[i].size());
                }

                d_subdomain_sizes.insert(d_subdomain_sizes.end(), h_subdomain_sizes.begin(), h_subdomain_sizes.end());

                factorize_submatrix();
            }

            size_t num_bad = 0;
            for (const auto &sub : bad_indices_arrays)
                num_bad += sub.size();
            info_["num_subdomains"] = bad_indices_arrays.size();
            info_["num_subdomain_dofs"] = num_bad;
            info_["subdomain_or_local_precond_setup_time"] = elapsed_seconds(phase_begin);
            SPDLOG_INFO("[{}] [setup_problematic_dof_precond] [{:.6f}]", name(), elapsed_seconds(phase_begin));
        }

        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
            A = nullptr;
        }

        copy_matrix_to_hypre();
        has_matrix_ = true;

        d_outer_indices.clear();
        d_inner_indices.clear();
        d_values.clear();
    }

    namespace
    {
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
            // print_level = 3;
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

    void GPUHybridSolver::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
    {
        num_iterations = 0;
        residual_history.clear();
        time_history.clear();

        thrust::device_vector<double> d_x(x.size());
        thrust::device_vector<double> d_b(b.size());

        thrust::copy(x.data(), x.data() + x.size(), d_x.begin());
        thrust::copy(b.data(), b.data() + b.size(), d_b.begin());

        if (direct_factorization != DirectFactorization::NONE)
        {
            auto phase_begin = clock::now();
            solve_direct(d_b, d_x);
            thrust::copy(d_x.begin(), d_x.end(), x.data());
            stop_reason = "direct";
            info_["direct_solve_time"] = elapsed_seconds(phase_begin);
            return;
        }

        HYPRE_ParVector par_b;
        HYPRE_ParVector par_x;
        init_hypre_vectors(b.size());

        set_hypre_vec(ij_b, par_b, d_b);
        set_hypre_vec(ij_x, par_x, d_x);

        // BoomerAMG (HYBRID), hypre ILU (ILU), or none here (JACOBI and IC
        // were set up in factorize)
        HYPRE_Solver precond = nullptr;

        if (preconditioner_type == PreconditionerType::ILU)
        {
            auto phase_begin = clock::now();
            HYPRE_ILUCreate(&precond);
            HYPRE_ILUSetType(precond, 0); // block-Jacobi ILU(k); one block on one GPU
            HYPRE_ILUSetLevelOfFill(precond, ilu_level_of_fill);
            HYPRE_ILUSetMaxIter(precond, 1);
            HYPRE_ILUSetTol(precond, 0.0);
            HYPRE_ILUSetPrintLevel(precond, 0);
            HYPRE_ILUSetup(precond, parcsr_A, par_b, par_x);
            CHECK_CUDA(cudaDeviceSynchronize());
            info_["precond_setup_time"] = elapsed_seconds(phase_begin);
            SPDLOG_INFO("[{}] [ilu_setup] [{:.6f}]", name(), elapsed_seconds(phase_begin));
        }
        else if (preconditioner_type != PreconditionerType::HYBRID)
        {
            info_["precond_setup_time"] = 0.0;
        }
        else
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
            info_["precond_setup_time"] = elapsed_seconds(phase_begin);
            SPDLOG_INFO("[{}] [amg_setup] [{:.6f}]", name(), elapsed_seconds(phase_begin));

            // Hierarchy size: levels and operator complexity (sum of nnz over
            // all levels / nnz of A).
            {
                hypre_ParAMGData *amg_data = (hypre_ParAMGData *)precond;
                const int num_levels = hypre_ParAMGDataNumLevels(amg_data);
                hypre_ParCSRMatrix **A_array = hypre_ParAMGDataAArray(amg_data);
                double nnz_total = 0.0, nnz_fine = 0.0;
                for (int level = 0; level < num_levels; ++level)
                {
                    hypre_ParCSRMatrixSetDNumNonzeros(A_array[level]);
                    const double nnz_level = hypre_ParCSRMatrixDNumNonzeros(A_array[level]);
                    nnz_total += nnz_level;
                    if (level == 0)
                        nnz_fine = nnz_level;
                }
                info_["amg_num_levels"] = num_levels;
                info_["amg_operator_complexity"] = nnz_fine > 0 ? nnz_total / nnz_fine : 0.0;
            }
        }

        {
            auto phase_begin = clock::now();

            if (krylov_method == KrylovMethod::GMRES)
                gmres_solve(d_b, d_x, precond);
            else
                pcg_solve(d_b, d_x, par_b, par_x, precond);
            CHECK_CUDA(cudaDeviceSynchronize());
            info_["krylov_time"] = elapsed_seconds(phase_begin);

            thrust::device_vector<double> buffer(d_x.size());
            matmul(d_x, buffer);
            vector_add(-1.0, d_b, buffer);
            final_res_norm = sqrt(dot(buffer, buffer));

            thrust::copy(d_x.begin(), d_x.end(), x.data());

            CHECK_CUDA(cudaDeviceSynchronize());
            SPDLOG_INFO("[{}] [krylov_solve] [{:.6f}] [iters={}] [residual={}]", name(), elapsed_seconds(phase_begin), num_iterations, final_res_norm);
        }

        {
            if (precond != nullptr)
            {
                if (preconditioner_type == PreconditionerType::ILU)
                    HYPRE_ILUDestroy(precond);
                else
                    HYPRE_BoomerAMGDestroy(precond);
            }
            HYPRE_IJVectorDestroy(ij_x);
            HYPRE_IJVectorDestroy(ij_b);
        }
    }

    void GPUHybridSolver::copy_matrix_to_hypre()
    {
        auto phase_begin = clock::now();

        const HYPRE_Int num_rows = d_outer_indices.size() - 1;
        const HYPRE_Int nnz = d_values.size();

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

    void GPUHybridSolver::init_hypre_vectors(const int size)
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

    void GPUHybridSolver::matmul(const thrust::device_vector<double> &x, thrust::device_vector<double> &result)
    {
        auto phase_begin = clock::now();
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_result;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_result, result);

        HYPRE_ParCSRMatrixMatvec(1.0, parcsr_A, par_x, 0.0, par_result);
        CHECK_CUDA(cudaDeviceSynchronize());
        GPUHYBRID_LOG_ITER("[{}] [matmul] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    double GPUHybridSolver::dot(const thrust::device_vector<double> &a, const thrust::device_vector<double> &b)
    {
        HYPRE_ParVector par_a;
        HYPRE_ParVector par_b;

        set_hypre_vec(ij_x, par_a, a);
        set_hypre_vec(ij_b, par_b, b);

        double result;
        HYPRE_ParVectorInnerProd(par_a, par_b, &result);
        return result;
    }

    void GPUHybridSolver::vector_copy(const thrust::device_vector<double> &x, thrust::device_vector<double> &y)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_y;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_y, y);

        HYPRE_ParVectorCopy(par_x, par_y);
    }

    void GPUHybridSolver::vector_add(double alpha, const thrust::device_vector<double> &x, thrust::device_vector<double> &y)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_y;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_y, y);

        hypre_ParVectorAxpy(alpha, par_x, par_y);
    }

    void GPUHybridSolver::vector_scale(double alpha, thrust::device_vector<double> &x)
    {
        HYPRE_ParVector par_x;

        set_hypre_vec(ij_x, par_x, x);

        HYPRE_ParVectorScale(alpha, par_x);
    }

    void GPUHybridSolver::set_hypre_vec(HYPRE_IJVector &my_ij_x, HYPRE_ParVector &par_x, const thrust::device_vector<double> &x)
    {
        double *raw_ptr = const_cast<double *>(thrust::raw_pointer_cast(x.data()));

        HYPRE_IJVectorSetData(my_ij_x, raw_ptr);
        HYPRE_IJVectorAssemble(my_ij_x);
        HYPRE_IJVectorGetObject(my_ij_x, (void **)&par_x);
    }

    void GPUHybridSolver::custom_mixed_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double> &r, thrust::device_vector<double> &z, thrust::device_vector<double> &buffer, thrust::device_vector<double> &z2)
    {
        if (d_all_bad_dofs.size() == 0 && num_contact_patches_ == 0)
        {
            amg_precond_iter(precond, r, z);
        }
        else if (additive_mode)
        {
            thrust::fill(buffer.begin(), buffer.end(), 0.0);
            thrust::fill(z2.begin(), z2.end(), 0.0);
            amg_precond_iter(precond, r, z);
            dss_precond_iter(buffer, r, z2);
            vector_add(1.0, z2, z);
        }
        else
        {
            thrust::fill(buffer.begin(), buffer.end(), 0.0);
            thrust::fill(z2.begin(), z2.end(), 0.0);
            amg_precond_iter(precond, r, z);
            dss_precond_iter(z, r, z2);
            matmul(z2, z);
            vector_copy(r, buffer);
            vector_add(-1.0, z, buffer);
            thrust::fill(z.begin(), z.end(), 0.0);
            amg_precond_iter(precond, buffer, z);
            vector_add(1.0, z2, z);
        }
    }

    void GPUHybridSolver::dss_precond_iter(thrust::device_vector<double> &z, thrust::device_vector<double> &r, thrust::device_vector<double> &next_z)
    {
        if (contact_patch_schwarz)
        {
            dense_contact_patch_precond_iter(z, r, next_z);
            return;
        }

        auto phase_begin = clock::now();

        matmul(z, next_z);
        vector_scale(-1.0, next_z);
        vector_add(1.0, r, next_z);

        if (sparse_batch_count > 0)
        {
            thrust::gather(
                thrust::device,
                d_sparse_dof_map.begin(),
                d_sparse_dof_map.end(),
                next_z.begin(),
                d_sparse_b.begin());

            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, cudss_config, cudss_solver_data, batch_A, batch_x, batch_b));
        }

        thrust::fill(next_z.begin(), next_z.end(), 0.0);

        if (sparse_batch_count > 0)
        {
            thrust::scatter(
                thrust::device,
                d_sparse_x.begin(),
                d_sparse_x.begin() + d_sparse_dof_map.size(),
                d_sparse_dof_map.begin(),
                next_z.begin());
        }

        vector_add(1.0, z, next_z);

        CHECK_CUDA(cudaDeviceSynchronize());
        GPUHYBRID_LOG_ITER("[{}] [subdomain_solve] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    void GPUHybridSolver::dense_contact_patch_precond_iter(thrust::device_vector<double> &z, thrust::device_vector<double> &r, thrust::device_vector<double> &next_z)
    {
        auto phase_begin = clock::now();

        matmul(z, next_z);
        vector_scale(-1.0, next_z);
        vector_add(1.0, r, next_z);

        if (num_contact_patches_ > 0)
        {
            thrust::fill(d_dense_b.begin(), d_dense_b.end(), 0.0);

            const double *raw_next_z = thrust::raw_pointer_cast(next_z.data());
            double *raw_dense_b = thrust::raw_pointer_cast(d_dense_b.data());
            const int *raw_dof_map = thrust::raw_pointer_cast(d_contact_dof_map.data());
            const int *raw_slot = thrust::raw_pointer_cast(d_contact_real_to_dense_slot.data());

            // Scatter each real dof's current rhs into its patch's padded
            // slot; padding slots stay 0 from the fill above, so they solve
            // to 0 and never affect the real dofs (see factorize_contact_patches_dense).
            thrust::for_each(thrust::device,
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(total_contact_patch_real_dofs_),
                             [=] __device__(int i) {
                                 raw_dense_b[raw_slot[i]] = raw_next_z[raw_dof_map[i]];
                             });

            // getrsBatched's info is a single HOST scalar (parameter validation
            // only), unlike getrfBatched's per-matrix DEVICE info array used in
            // factorize_contact_patches_dense -- these are not interchangeable.
            int host_info = 0;
            CHECK_CUBLAS(cublasDgetrsBatched(
                cublas_handle, CUBLAS_OP_N, kContactPatchDenseSize, 1,
                thrust::raw_pointer_cast(d_dense_A_ptrs.data()), kContactPatchDenseSize,
                thrust::raw_pointer_cast(d_dense_pivot.data()),
                thrust::raw_pointer_cast(d_dense_b_ptrs.data()), kContactPatchDenseSize,
                &host_info, num_contact_patches_));
            if (host_info != 0)
            {
                throw std::runtime_error("GPUHybridSolver: cublasDgetrsBatched reported an invalid argument (info=" + std::to_string(host_info) + ")");
            }
        }

        thrust::fill(next_z.begin(), next_z.end(), 0.0);

        if (num_contact_patches_ > 0)
        {
            // Contact patches may overlap, so a dof's solved value can come
            // from more than one patch. Gather every real entry's solved
            // value (still in the padded per-patch layout) into the order
            // sorted by global dof -- precomputed once at factorize time --
            // then sum each dof's duplicate entries with reduce_by_key
            // instead of racing to scatter overlapping writes into next_z.
            const double *raw_dense_b = thrust::raw_pointer_cast(d_dense_b.data());
            const int *raw_slot = thrust::raw_pointer_cast(d_contact_real_to_dense_slot.data());
            const int *raw_perm = thrust::raw_pointer_cast(d_reduce_perm.data());
            double *raw_reduce_values = thrust::raw_pointer_cast(d_reduce_values.data());

            thrust::for_each(thrust::device,
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(total_contact_patch_real_dofs_),
                             [=] __device__(int k) {
                                 int i = raw_perm[k];
                                 raw_reduce_values[k] = raw_dense_b[raw_slot[i]];
                             });

            thrust::reduce_by_key(
                thrust::device,
                d_reduce_keys.begin(), d_reduce_keys.end(),
                d_reduce_values.begin(),
                d_reduce_unique_keys.begin(), d_reduce_unique_values.begin());

            const double scale = overlap_scale_;
            thrust::transform(thrust::device,
                              d_reduce_unique_values.begin(), d_reduce_unique_values.begin() + num_unique_overlap_dofs_,
                              d_reduce_unique_values.begin(),
                              [scale] __device__(double v) { return v * scale; });

            thrust::scatter(
                thrust::device,
                d_reduce_unique_values.begin(), d_reduce_unique_values.begin() + num_unique_overlap_dofs_,
                d_reduce_unique_keys.begin(),
                next_z.begin());
        }

        vector_add(1.0, z, next_z);

        CHECK_CUDA(cudaDeviceSynchronize());
        GPUHYBRID_LOG_ITER("[{}] [contact_patch_dense_subdomain_solve] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    void GPUHybridSolver::amg_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double> &b, thrust::device_vector<double> &x)
    {
        auto phase_begin = clock::now();
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_b;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_b, b);

        HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);
        CHECK_CUDA(cudaDeviceSynchronize());
        GPUHYBRID_LOG_ITER("[{}] [amg_v_cycle] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    void GPUHybridSolver::decompose_subdomains_to_disjoint_subsets(const Eigen::SparseMatrix<double> &sparse_A)
    {
        auto phase_begin = clock::now();

        std::vector<int> global_to_local(sparse_A.rows(), -1);
        int counter = 0;
        for (auto index : h_all_bad_dofs)
        {
            global_to_local[index] = counter;
            ++counter;
        }

        hybrid::DisjointSet decomposed_subdomains(h_all_bad_dofs.size());

        for (int k : h_all_bad_dofs)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_A, k); it; ++it)
            {
                if (global_to_local[it.row()] != -1)
                {
                    decomposed_subdomains.union_set(global_to_local[it.row()], global_to_local[it.col()]);
                }
            }
        }

        std::unordered_map<int, std::vector<int>> chosen_sets;
        for (auto index : h_all_bad_dofs)
        {
            chosen_sets[decomposed_subdomains.find_set(global_to_local[index])].push_back(index);
        }

        bad_indices_arrays.clear();

        for (auto &kv : chosen_sets)
        {
            if (kv.second.size() > max_subdomain_size)
            {
                continue;
            }
            bad_indices_arrays.emplace_back(kv.second.begin(), kv.second.end());
        }

        SPDLOG_INFO("[{}] [subdomain_decomposition] [{}] [num_subdomains={}] ",
                     name(), elapsed_seconds(phase_begin), bad_indices_arrays.size());
    }

    void GPUHybridSolver::select_bad_dofs()
    {
        if (!select_bad_dofs_from_l1_row_norm)
        {
            // AMGF mode: bad DOFs come only from whatever set_problematic_dofs
            // supplied, not from this heuristic.
            return;
        }

        auto phase_begin = clock::now();

        const int num_rows = d_outer_indices.size() - 1;

        const int *row_offsets = thrust::raw_pointer_cast(d_outer_indices.data());
        const double *values = thrust::raw_pointer_cast(d_values.data());

        thrust::device_vector<double> d_row_norms(num_rows);
        double *row_norms = thrust::raw_pointer_cast(d_row_norms.data());

        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(num_rows),
                         [=] __device__(int i) {
                             int start = row_offsets[i];
                             int end = row_offsets[i + 1];
                             double sum = 0.0;
                             for (int j = start; j < end; ++j)
                             {
                                 sum += fabs(values[j]);
                             }
                             row_norms[i] = sum;
                         });

        double sum_norms = thrust::reduce(d_row_norms.begin(), d_row_norms.end(), 0.0);
        double global_mean = sum_norms / num_rows;

        double var_sum = thrust::transform_reduce(thrust::device, d_row_norms.begin(), d_row_norms.end(), [global_mean] __device__(double x) -> double { return (x - global_mean) * (x - global_mean); }, 0.0, thrust::plus<double>());
        double global_var = var_sum / num_rows;

        h_all_bad_dofs.clear();

        double mean_0 = 0.0, mean_1 = 0.0;
        double var_0 = 0.0, var_1 = 0.0;
        int gmm_iter = 0;

        double max_dist = -1.0;
        double max_jump = -1.0;
        double min_cost = -1.0;
        int split_idx = 0;

        switch (subdomain_selection_strategy)
        {
        case SubdomainSelectionStrategy::KNEE:
        case SubdomainSelectionStrategy::FD:
        case SubdomainSelectionStrategy::COST:
        {
            // Sort on GPU, preserving original row indices
            thrust::device_vector<double> d_sorted_norms = d_row_norms;
            thrust::device_vector<int> d_indices(num_rows);
            thrust::sequence(d_indices.begin(), d_indices.end());

            thrust::sort_by_key(d_sorted_norms.begin(), d_sorted_norms.end(), d_indices.begin());

            // Bring sorted data back to host for O(N) linear scanning
            std::vector<double> h_sorted_norms(num_rows);
            std::vector<int> h_indices(num_rows);

            thrust::copy(d_sorted_norms.begin(), d_sorted_norms.end(), h_sorted_norms.begin());
            thrust::copy(d_indices.begin(), d_indices.end(), h_indices.begin());

            int N = num_rows;
            if (N >= 2)
            {
                if (subdomain_selection_strategy == SubdomainSelectionStrategy::KNEE)
                {
                    double x1 = 0.0;
                    double y1 = std::log(h_sorted_norms[0] + 1e-12);
                    double x2 = N - 1.0;
                    double y2 = std::log(h_sorted_norms[N - 1] + 1e-12);

                    for (int i = 1; i < N - 1; ++i)
                    {
                        double x0 = (double)i;
                        double y0 = std::log(h_sorted_norms[i] + 1e-12);

                        double dist = std::abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1);

                        if (dist > max_dist)
                        {
                            max_dist = dist;
                            split_idx = i;
                        }
                    }
                }
                else if (subdomain_selection_strategy == SubdomainSelectionStrategy::FD)
                {
                    for (int i = 1; i < N; ++i)
                    {
                        double jump = h_sorted_norms[i] - h_sorted_norms[i - 1];
                        if (jump > max_jump)
                        {
                            max_jump = jump;
                            split_idx = i;
                        }
                    }
                }
                else if (subdomain_selection_strategy == SubdomainSelectionStrategy::COST)
                {
                    const double c_pcg_conv = 1.0;
                    const double c_pcg = 1.0;
                    const double c_fac = 1.0;
                    const double c_solve = 1.0;

                    min_cost = std::numeric_limits<double>::max();

                    double lambda_min = h_sorted_norms[0];

                    for (int i = 0; i < N; ++i)
                    {
                        double p = static_cast<double>(N - i);
                        double lambda_max = h_sorted_norms[i];

                        double m = c_pcg_conv * std::sqrt(lambda_max / lambda_min);
                        double cost = (m * c_pcg * N) +
                                      (c_fac * p * p) +
                                      (m * c_solve * std::pow(p, 4.0 / 3.0));

                        if (cost < min_cost)
                        {
                            min_cost = cost;
                            split_idx = i;
                        }
                    }
                }

                std::vector<int> h_bad_dofs;
                h_bad_dofs.reserve(N - split_idx);

                for (int i = split_idx; i < N; ++i)
                {
                    h_all_bad_dofs.insert(h_indices[i]);
                    h_bad_dofs.push_back(h_indices[i]);
                }

                d_all_bad_dofs = h_bad_dofs;
            }
            break;
        }
        case SubdomainSelectionStrategy::GMM:
        {
            auto minmax = thrust::minmax_element(d_row_norms.begin(), d_row_norms.end());
            mean_0 = *minmax.first;
            mean_1 = *minmax.second;
            var_0 = global_var;
            var_1 = global_var;
            double w0 = 0.5;
            double w1 = 0.5;

            double var_reg = 1e-6;

            thrust::device_vector<double> d_gamma0(num_rows);
            thrust::device_vector<double> d_gamma1(num_rows);
            double *g0 = thrust::raw_pointer_cast(d_gamma0.data());
            double *g1 = thrust::raw_pointer_cast(d_gamma1.data());

            for (gmm_iter = 0; gmm_iter < max_gmm_iterations; ++gmm_iter)
            {
                double log_likelihood = thrust::transform_reduce(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_rows), [=] __device__(int i) -> double {
                        double x = row_norms[i];

                        double log_w0 = log(w0);
                        double log_w1 = log(w1);

                        double log_N0 = -0.5 * log(2.0 * M_PI * var_0) - 0.5 * (x - mean_0) * (x - mean_0) / var_0;
                        double log_N1 = -0.5 * log(2.0 * M_PI * var_1) - 0.5 * (x - mean_1) * (x - mean_1) / var_1;

                        double log_g0 = log_w0 + log_N0;
                        double log_g1 = log_w1 + log_N1;

                        double max_log_g = max(log_g0, log_g1);
                        double log_total = max_log_g + log(exp(log_g0 - max_log_g) + exp(log_g1 - max_log_g));

                        g0[i] = exp(log_g0 - log_total);
                        g1[i] = exp(log_g1 - log_total);

                        return log_total; }, 0.0, thrust::plus<double>());

                double sum_g0 = thrust::reduce(d_gamma0.begin(), d_gamma0.end(), 0.0);
                double sum_g1 = thrust::reduce(d_gamma1.begin(), d_gamma1.end(), 0.0);

                w0 = sum_g0 / num_rows;
                w1 = sum_g1 / num_rows;

                double old_mean_0 = mean_0, old_mean_1 = mean_1;
                double old_var_0 = var_0, old_var_1 = var_1;

                mean_0 = thrust::inner_product(d_gamma0.begin(), d_gamma0.end(), d_row_norms.begin(), 0.0) / sum_g0;
                mean_1 = thrust::inner_product(d_gamma1.begin(), d_gamma1.end(), d_row_norms.begin(), 0.0) / sum_g1;

                var_0 = thrust::transform_reduce(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_rows), [=] __device__(int i) -> double { return g0[i] * (row_norms[i] - mean_0) * (row_norms[i] - mean_0); }, 0.0, thrust::plus<double>()) / sum_g0 + var_reg;

                var_1 = thrust::transform_reduce(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_rows), [=] __device__(int i) -> double { return g1[i] * (row_norms[i] - mean_1) * (row_norms[i] - mean_1); }, 0.0, thrust::plus<double>()) / sum_g1 + var_reg;

                // Check Convergence
                if (abs(mean_0 - old_mean_0) / abs(old_mean_0) < gmm_tol && abs(mean_1 - old_mean_1) / abs(old_mean_1) < gmm_tol && abs(var_0 - old_var_0) / abs(old_var_0) < gmm_tol && abs(var_1 - old_var_1) / abs(old_var_1) < gmm_tol)
                {
                    break;
                }
            }

            int num_bad_dofs = 0;
            if (abs(mean_1) / abs(mean_0) > gmm_jump_threshold)
            {
                d_all_bad_dofs.resize(num_rows);
                auto end_it = thrust::copy_if(thrust::device,
                                              thrust::make_counting_iterator(0),
                                              thrust::make_counting_iterator(num_rows),
                                              d_all_bad_dofs.begin(),
                                              [=] __device__(int i) { return g0[i] < g1[i]; });

                num_bad_dofs = thrust::distance(d_all_bad_dofs.begin(), end_it);
            }

            d_all_bad_dofs.resize(num_bad_dofs);

            std::vector<int> h_bad_dofs(num_bad_dofs);
            thrust::copy(d_all_bad_dofs.begin(), d_all_bad_dofs.end(), h_bad_dofs.begin());

            h_all_bad_dofs.insert(h_bad_dofs.begin(), h_bad_dofs.end());
            break;
        }
        case SubdomainSelectionStrategy::APOSTERIORI:
            break;
        }

        switch (subdomain_selection_strategy)
        {
        case SubdomainSelectionStrategy::KNEE:
            SPDLOG_INFO("[{}] [bad_dof_selection] [{:.6f}] [strategy=KNEE] [global_mean={}] [global_var={}] [split_idx={}] [max_dist={}] [num_bad_dofs={}]",
                         name(), elapsed_seconds(phase_begin), global_mean, global_var, split_idx, max_dist, h_all_bad_dofs.size());
            break;
        case SubdomainSelectionStrategy::FD:
            SPDLOG_INFO("[{}] [bad_dof_selection] [{:.6f}] [strategy=FD] [global_mean={}] [global_var={}] [split_idx={}] [max_jump={}] [num_bad_dofs={}]",
                         name(), elapsed_seconds(phase_begin), global_mean, global_var, split_idx, max_jump, h_all_bad_dofs.size());
            break;
        case SubdomainSelectionStrategy::COST:
            SPDLOG_INFO("[{}] [bad_dof_selection] [{:.6f}] [strategy=COST] [global_mean={}] [global_var={}] [split_idx={}] [min_cost={}] [num_bad_dofs={}]",
                         name(), elapsed_seconds(phase_begin), global_mean, global_var, split_idx, min_cost, h_all_bad_dofs.size());
            break;
        case SubdomainSelectionStrategy::GMM:
            SPDLOG_INFO("[{}] [bad_dof_selection] [{:.6f}] [strategy=GMM] [global_mean={}] [global_var={}] [mean_0={}] [mean_1={}] [var_0={}] [var_1={}] [gmm_iters={}] [num_bad_dofs={}]",
                         name(), elapsed_seconds(phase_begin), global_mean, global_var, mean_0, mean_1, var_0, var_1, gmm_iter, h_all_bad_dofs.size());
            break;
        case SubdomainSelectionStrategy::APOSTERIORI:
            break;
        }
    }

    void GPUHybridSolver::filter_subdomains(const Eigen::SparseMatrix<double> &sparse_A)
    {
        auto phase_begin = clock::now();

        int num_too_small = 0;
        int num_too_large = 0;
        int num_not_poorly_conditioned = 0;
        int original_num_bad_dofs = h_all_bad_dofs.size();

        int counter = 0;
        std::vector<int> global_to_local(sparse_A.rows(), -1);
        for (auto index : h_all_bad_dofs)
        {
            global_to_local[index] = counter;
            ++counter;
        }

        hybrid::DisjointSet decomposed_subdomains(h_all_bad_dofs.size());

        for (int k : h_all_bad_dofs)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_A, k); it; ++it)
            {
                if (global_to_local[it.row()] != -1)
                {
                    decomposed_subdomains.union_set(global_to_local[it.row()], global_to_local[it.col()]);
                }
            }
        }

        std::unordered_map<int, std::vector<int>> chosen_sets;
        for (auto index : h_all_bad_dofs)
        {
            chosen_sets[decomposed_subdomains.find_set(global_to_local[index])].push_back(index);
        }

        h_all_bad_dofs.clear();

        for (auto &kv : chosen_sets)
        {
            if (kv.second.size() < min_subdomain_size)
            {
                ++num_too_small;
                continue;
            }
            if (kv.second.size() > max_subdomain_size)
            {
                ++num_too_large;
                continue;
            }

            double lambda_min = std::numeric_limits<double>::max();
            double lambda_max = 0.0;
            for (int k : kv.second)
            {
                double diag_value = 0.0;
                double abs_off_diag_sum = 0.0;
                for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_A, k); it; ++it)
                {
                    if (global_to_local[it.row()] != -1)
                    {
                        if (it.row() == it.col())
                        {
                            diag_value = it.value();
                        }
                        else
                        {
                            abs_off_diag_sum += abs(it.value());
                        }
                    }
                }
                lambda_min = std::min(lambda_min, diag_value - abs_off_diag_sum);
                lambda_max = std::max(lambda_max, diag_value + abs_off_diag_sum);
            }

            if (lambda_min * lambda_max < 0.0 || lambda_max / lambda_min > conditioning_threshold)
            {
                h_all_bad_dofs.insert(kv.second.begin(), kv.second.end());
                continue;
            }
            ++num_not_poorly_conditioned;
        }

        SPDLOG_INFO("[{}] [subdomain_filtering] [{}] [total_dofs_before={}] [total_dofs_after={}] [num_too_small={}] [num_too_large={}] [num_not_poorly_conditioned={}]",
                     name(), elapsed_seconds(phase_begin), original_num_bad_dofs, h_all_bad_dofs.size(), num_too_small, num_too_large, num_not_poorly_conditioned);
    }

    void GPUHybridSolver::expand_subdomains_to_strongly_connected(const Eigen::SparseMatrix<double> &sparse_A)
    {
        auto phase_begin = clock::now();
        int num_bad_dofs_before = h_all_bad_dofs.size();

        std::set<int> new_bad_dofs;
        ;

        for (int k : h_all_bad_dofs)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_A, k); it; ++it)
            {
                new_bad_dofs.insert(it.row());
            }
        }

        h_all_bad_dofs = std::move(new_bad_dofs);

        SPDLOG_INFO("[{}] [subdomain_expansion] [{}] [num_dofs_before={}] [num_dofs_after={}]",
                     name(), elapsed_seconds(phase_begin), num_bad_dofs_before, h_all_bad_dofs.size());
    }

    // Under contact_patch_schwarz, bad_indices_arrays (the contact patches)
    // may overlap, so a covered dof can belong to more than one patch.
    // Compute a single global scale factor -- 1 / (average number of patches
    // per covered dof) -- used to damp the summed correction in
    // dense_contact_patch_precond_iter so overlapping patches don't
    // over-correct on average. Also records the number of distinct covered
    // dofs, needed to size the reduce_by_key output in that same function.
    void GPUHybridSolver::compute_overlap_scale()
    {
        std::unordered_map<int, int> dof_patch_count;
        for (const auto &subdomain : bad_indices_arrays)
        {
            for (int dof : subdomain)
            {
                ++dof_patch_count[dof];
            }
        }

        num_unique_overlap_dofs_ = static_cast<int>(dof_patch_count.size());

        if (dof_patch_count.empty())
        {
            overlap_scale_ = 1.0;
            return;
        }

        long long total_incidences = 0;
        for (const auto &kv : dof_patch_count)
        {
            total_incidences += kv.second;
        }

        const double avg_overlap = static_cast<double>(total_incidences) / static_cast<double>(dof_patch_count.size());
        overlap_scale_ = 1.0 / avg_overlap;

        SPDLOG_INFO("[{}] [contact_patch_overlap] [avg_overlap={}] [overlap_scale={}] [num_unique_dofs={}]",
                     name(), avg_overlap, overlap_scale_, num_unique_overlap_dofs_);
    }

    void GPUHybridSolver::factorize_contact_patches_dense(const Eigen::SparseMatrix<double> &sparse_A)
    {
        auto phase_begin = clock::now();

        free_contact_patch_dense_memory();

        constexpr int N = kContactPatchDenseSize;

        num_contact_patches_ = bad_indices_arrays.size();

        for (const auto &patch : bad_indices_arrays)
        {
            if ((int)patch.size() > N)
            {
                throw std::runtime_error(
                    "GPUHybridSolver: contact patch has " + std::to_string(patch.size()) +
                    " dofs, which exceeds the fixed dense batched solve size of " + std::to_string(N));
            }
            total_contact_patch_real_dofs_ += patch.size();
        }

        if (num_contact_patches_ == 0)
        {
            SPDLOG_INFO("[{}] [factorize_contact_patches_dense] [{:.6f}] [n_patches=0]", name(), elapsed_seconds(phase_begin));
            return;
        }

        // Assemble each patch's dense (padded) block on the host from the
        // sparse system matrix, the same way the CPU hybrid solver's
        // assemble_D does. This runs once per factorize() call, not on the
        // per-iteration hot path, so host-side assembly is fine here even
        // though the factorization and every solve happen on the GPU.
        std::vector<double> h_dense_A(static_cast<size_t>(num_contact_patches_) * N * N, 0.0);
        std::vector<int> h_dof_map;
        std::vector<int> h_real_to_dense_slot;
        h_dof_map.reserve(total_contact_patch_real_dofs_);
        h_real_to_dense_slot.reserve(total_contact_patch_real_dofs_);

        for (int p = 0; p < num_contact_patches_; ++p)
        {
            const auto &patch = bad_indices_arrays[p];
            std::unordered_map<int, int> local_index;
            local_index.reserve(patch.size());
            for (int i = 0; i < (int)patch.size(); ++i)
            {
                local_index[patch[i]] = i;
            }

            double *block = h_dense_A.data() + (size_t)p * N * N;

            for (int local_col = 0; local_col < (int)patch.size(); ++local_col)
            {
                int global_col = patch[local_col];
                for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_A, global_col); it; ++it)
                {
                    auto found = local_index.find(it.row());
                    if (found != local_index.end())
                    {
                        // Column-major (lda = N), to match cublas<t>getrfBatched/getrsBatched.
                        block[found->second + local_col * N] = it.value();
                    }
                }

                h_dof_map.push_back(global_col);
                h_real_to_dense_slot.push_back(p * N + local_col);
            }

            // Pad the unused rows/cols with an identity block, so the padded
            // system stays nonsingular and decouples from the real dofs --
            // the padded slots always solve to 0, leaving the real dofs'
            // solve exactly as if the patch had been solved unpadded.
            for (int i = (int)patch.size(); i < N; ++i)
            {
                block[i + i * N] = 1.0;
            }
        }

        d_dense_A = h_dense_A;
        d_dense_pivot.resize((size_t)num_contact_patches_ * N);
        d_dense_info.resize(num_contact_patches_);
        d_dense_b.resize((size_t)num_contact_patches_ * N);
        d_contact_dof_map = h_dof_map;
        d_contact_real_to_dense_slot = h_real_to_dense_slot;

        std::vector<double *> h_dense_A_ptrs(num_contact_patches_);
        std::vector<double *> h_dense_b_ptrs(num_contact_patches_);
        double *raw_dense_A = thrust::raw_pointer_cast(d_dense_A.data());
        double *raw_dense_b = thrust::raw_pointer_cast(d_dense_b.data());
        for (int p = 0; p < num_contact_patches_; ++p)
        {
            h_dense_A_ptrs[p] = raw_dense_A + (size_t)p * N * N;
            h_dense_b_ptrs[p] = raw_dense_b + (size_t)p * N;
        }
        d_dense_A_ptrs = h_dense_A_ptrs;
        d_dense_b_ptrs = h_dense_b_ptrs;

        // Factorize every patch's padded block in one batched call. d_dense_info
        // is a device array (one entry per patch), per cublasDgetrfBatched's
        // documented convention -- unlike getrsBatched's info below, which is
        // a single host scalar.
        CHECK_CUBLAS(cublasDgetrfBatched(
            cublas_handle, N,
            thrust::raw_pointer_cast(d_dense_A_ptrs.data()), N,
            thrust::raw_pointer_cast(d_dense_pivot.data()),
            thrust::raw_pointer_cast(d_dense_info.data()),
            num_contact_patches_));
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<int> h_info(num_contact_patches_);
        thrust::copy(d_dense_info.begin(), d_dense_info.end(), h_info.begin());
        for (int p = 0; p < num_contact_patches_; ++p)
        {
            if (h_info[p] != 0)
            {
                SPDLOG_WARN("[{}] [contact_patch_singular] [patch={}] [info={}]", name(), p, h_info[p]);
            }
        }

        compute_overlap_scale();

        // Precompute the sort-by-global-dof plan used every iteration to sum
        // overlapping patches' contributions to a shared dof with
        // thrust::reduce_by_key, instead of racing to scatter into it. Since
        // bad_indices_arrays (and so d_contact_dof_map) is fixed for the life
        // of this factorization, both the permutation and the sorted keys
        // only need to be computed once, here.
        d_reduce_perm.resize(total_contact_patch_real_dofs_);
        thrust::sequence(d_reduce_perm.begin(), d_reduce_perm.end());

        d_reduce_keys = d_contact_dof_map;
        thrust::sort_by_key(d_reduce_keys.begin(), d_reduce_keys.end(), d_reduce_perm.begin());

        d_reduce_values.resize(total_contact_patch_real_dofs_);
        d_reduce_unique_keys.resize(total_contact_patch_real_dofs_);
        d_reduce_unique_values.resize(total_contact_patch_real_dofs_);

        SPDLOG_INFO("[{}] [factorize_contact_patches_dense] [{:.6f}] [n_patches={}] [n_real_dofs={}] [n_unique_dofs={}]",
                     name(), elapsed_seconds(phase_begin), num_contact_patches_, total_contact_patch_real_dofs_, num_unique_overlap_dofs_);
    }

    void GPUHybridSolver::factorize_submatrix()
    {
        auto phase_begin = clock::now();

        if (d_all_bad_dofs.size() == 0)
        {
            return;
        }

        free_device_memory();

        CHECK_CUDSS(cudssConfigCreate(&cudss_config));
        CHECK_CUDSS(cudssDataCreate(cudss_handle, &cudss_solver_data));

        int total_sparse_dofs = 0;
        std::vector<int> h_sparse_row_starts;
        std::vector<int> h_sparse_nnz_starts;
        std::vector<int> h_sparse_dof_starts;
        h_sparse_row_starts.push_back(0);
        h_sparse_dof_starts.push_back(0);
        h_sparse_nrows.clear();
        h_sparse_ncols.clear();
        h_sparse_vec_ncols.clear();
        h_sparse_ld.clear();
        for (int size : h_subdomain_sizes)
        {
            h_sparse_nrows.push_back(size);
            h_sparse_ncols.push_back(size);
            h_sparse_vec_ncols.push_back(1);
            h_sparse_ld.push_back(size);

            h_sparse_row_starts.push_back(h_sparse_row_starts.back() + size + 1);
            h_sparse_dof_starts.push_back(h_sparse_dof_starts.back() + size);

            total_sparse_dofs += size;
        }

        sparse_batch_count = h_sparse_nrows.size();

        d_sparse_dof_map.clear();
        d_sparse_dof_map.reserve(total_sparse_dofs);

        if (total_sparse_dofs > 0)
        {
            std::vector<int> h_sparse_dof_map;

            for (int i = 0; i < bad_indices_arrays.size(); ++i)
            {
                h_sparse_dof_map.insert(h_sparse_dof_map.end(), bad_indices_arrays[i].begin(), bad_indices_arrays[i].end());
            }

            d_sparse_dof_map.insert(d_sparse_dof_map.begin(), h_sparse_dof_map.begin(), h_sparse_dof_map.end());

            h_sparse_nnz.clear();
            h_sparse_nnz.resize(sparse_batch_count);

            thrust::device_vector<int> d_sparse_nnz(sparse_batch_count, 0);
            thrust::device_vector<int> d_sparse_batch_offsets(h_sparse_dof_starts.begin(), h_sparse_dof_starts.end());
            thrust::device_vector<int> d_sparse_batch_sizes(h_sparse_nrows.begin(), h_sparse_nrows.end());

            int total_sparse_dofs = d_sparse_dof_map.size();

            int *raw_bad_dofs = thrust::raw_pointer_cast(d_sparse_dof_map.data());
            int *raw_batch_offsets = thrust::raw_pointer_cast(d_sparse_batch_offsets.data());
            int *raw_batch_sizes = thrust::raw_pointer_cast(d_sparse_batch_sizes.data());
            int *raw_outer = thrust::raw_pointer_cast(d_outer_indices.data());
            int *raw_inner = thrust::raw_pointer_cast(d_inner_indices.data());

            thrust::device_vector<int> d_row_nnz(total_sparse_dofs, 0);
            int *raw_row_nnz = thrust::raw_pointer_cast(d_row_nnz.data());

            thrust::fill(d_sparse_nnz.begin(), d_sparse_nnz.end(), 0);
            int *raw_sparse_nnz = thrust::raw_pointer_cast(d_sparse_nnz.data());

            int local_sparse_batch_count = sparse_batch_count;

            thrust::for_each(thrust::device,
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(total_sparse_dofs),
                             [=] __device__(int i) {
                                 int *batch_ptr = thrust::upper_bound(thrust::seq, raw_batch_offsets, raw_batch_offsets + local_sparse_batch_count, i);
                                 int batch_idx = (batch_ptr - raw_batch_offsets) - 1;

                                 int offset = raw_batch_offsets[batch_idx];
                                 int size = raw_batch_sizes[batch_idx];

                                 int global_row = raw_bad_dofs[i];
                                 int row_start = raw_outer[global_row];
                                 int row_end = raw_outer[global_row + 1];

                                 int *sub_begin = raw_bad_dofs + offset;
                                 int *sub_end = sub_begin + size;

                                 int row_nnz = 0;
                                 for (int j = row_start; j < row_end; ++j)
                                 {
                                     int global_col = raw_inner[j];
                                     if (thrust::binary_search(thrust::seq, sub_begin, sub_end, global_col))
                                     {
                                         row_nnz++;
                                     }
                                 }

                                 raw_row_nnz[i] = row_nnz;
                                 atomicAdd(&raw_sparse_nnz[batch_idx], row_nnz);
                             });

            thrust::copy(d_sparse_nnz.begin(), d_sparse_nnz.end(), h_sparse_nnz.begin());
            h_sparse_nnz_starts.resize(h_sparse_nnz.size());
            h_sparse_nnz_starts[0] = 0;
            std::partial_sum(h_sparse_nnz.begin(), h_sparse_nnz.end() - 1, h_sparse_nnz_starts.begin() + 1);
            int total_sparse_nnz = std::accumulate(h_sparse_nnz.begin(), h_sparse_nnz.end(), 0);
            d_sparse_inner_indices.clear();
            d_sparse_outer_indices.clear();
            d_sparse_values.clear();
            d_sparse_x.clear();
            d_sparse_b.clear();

            d_sparse_outer_indices.resize(total_sparse_dofs + sparse_batch_count);
            d_sparse_inner_indices.resize(total_sparse_nnz);

            d_sparse_x.resize(total_sparse_dofs);
            d_sparse_b.resize(total_sparse_dofs);
            d_sparse_values.resize(total_sparse_nnz);

            thrust::device_vector<int> d_row_nnz_starts(total_sparse_dofs, 0);
            thrust::exclusive_scan(thrust::device, d_row_nnz.begin(), d_row_nnz.end(), d_row_nnz_starts.begin());

            int *raw_row_nnz_starts = thrust::raw_pointer_cast(d_row_nnz_starts.data());
            int *raw_sparse_outer = thrust::raw_pointer_cast(d_sparse_outer_indices.data());
            int *raw_sparse_inner = thrust::raw_pointer_cast(d_sparse_inner_indices.data());

            double *raw_global_values = thrust::raw_pointer_cast(d_values.data());
            double *raw_sparse_values = thrust::raw_pointer_cast(d_sparse_values.data());

            thrust::for_each(thrust::device,
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(total_sparse_dofs),
                             [=] __device__(int i) {
                                 int *batch_ptr = thrust::upper_bound(thrust::seq, raw_batch_offsets, raw_batch_offsets + local_sparse_batch_count, i);
                                 int batch_idx = (batch_ptr - raw_batch_offsets) - 1;

                                 int offset = raw_batch_offsets[batch_idx];
                                 int size = raw_batch_sizes[batch_idx];

                                 int r = i - offset;
                                 int outer_start = offset + batch_idx;

                                 if (r == 0)
                                 {
                                     raw_sparse_outer[outer_start] = 0;
                                 }

                                 int batch_nnz_start = raw_row_nnz_starts[offset];

                                 raw_sparse_outer[outer_start + r + 1] = (raw_row_nnz_starts[i] + raw_row_nnz[i]) - batch_nnz_start;

                                 int current_nnz = raw_row_nnz_starts[i];

                                 int global_row = raw_bad_dofs[i];
                                 int row_start = raw_outer[global_row];
                                 int row_end = raw_outer[global_row + 1];

                                 int *sub_begin = raw_bad_dofs + offset;
                                 int *sub_end = sub_begin + size;

                                 for (int j = row_start; j < row_end; ++j)
                                 {
                                     int global_col = raw_inner[j];

                                     int *ptr = thrust::lower_bound(thrust::seq, sub_begin, sub_end, global_col);

                                     if (ptr != sub_end && *ptr == global_col)
                                     {
                                         int local_col = ptr - sub_begin;

                                         raw_sparse_inner[current_nnz] = local_col;
                                         raw_sparse_values[current_nnz] = raw_global_values[j];

                                         current_nnz++;
                                     }
                                 }
                             });

            std::vector<void *> h_sparse_outer_void;
            std::vector<void *> h_sparse_inner_void;
            std::vector<void *> h_sparse_values_void;
            std::vector<void *> h_sparse_x_void;
            std::vector<void *> h_sparse_b_void;

            for (int i = 0; i < sparse_batch_count; ++i)
            {
                h_sparse_outer_void.push_back(static_cast<void *>(thrust::raw_pointer_cast(d_sparse_outer_indices.data()) + h_sparse_row_starts[i]));
                h_sparse_inner_void.push_back(static_cast<void *>(thrust::raw_pointer_cast(d_sparse_inner_indices.data()) + h_sparse_nnz_starts[i]));
                h_sparse_x_void.push_back(static_cast<void *>(thrust::raw_pointer_cast(d_sparse_x.data()) + h_sparse_dof_starts[i]));
                h_sparse_b_void.push_back(static_cast<void *>(thrust::raw_pointer_cast(d_sparse_b.data()) + h_sparse_dof_starts[i]));
                h_sparse_values_void.push_back(static_cast<void *>(thrust::raw_pointer_cast(d_sparse_values.data()) + h_sparse_nnz_starts[i]));
            }

            d_sparse_outer_void = h_sparse_outer_void;
            d_sparse_inner_void = h_sparse_inner_void;
            d_sparse_values_void = h_sparse_values_void;
            d_sparse_x_void = h_sparse_x_void;
            d_sparse_b_void = h_sparse_b_void;

            auto precision = CUDA_R_64F;

            CHECK_CUDSS(cudssMatrixCreateBatchDn(
                &batch_x, sparse_batch_count, h_sparse_nrows.data(), h_sparse_vec_ncols.data(), h_sparse_ld.data(),
                thrust::raw_pointer_cast(d_sparse_x_void.data()), CUDA_R_32I, precision, CUDSS_LAYOUT_COL_MAJOR));

            CHECK_CUDSS(cudssMatrixCreateBatchDn(
                &batch_b, sparse_batch_count, h_sparse_nrows.data(), h_sparse_vec_ncols.data(), h_sparse_ld.data(),
                thrust::raw_pointer_cast(d_sparse_b_void.data()), CUDA_R_32I, precision, CUDSS_LAYOUT_COL_MAJOR));

            CHECK_CUDSS(cudssMatrixCreateBatchCsr(
                &batch_A, sparse_batch_count, h_sparse_nrows.data(), h_sparse_ncols.data(), h_sparse_nnz.data(),
                thrust::raw_pointer_cast(d_sparse_outer_void.data()), nullptr, thrust::raw_pointer_cast(d_sparse_inner_void.data()), thrust::raw_pointer_cast(d_sparse_values_void.data()),
                CUDA_R_32I, precision,
                // GMRES may be given a nonsymmetric matrix: LU on its subdomains.
                krylov_method == KrylovMethod::GMRES ? CUDSS_MTYPE_GENERAL : CUDSS_MTYPE_SYMMETRIC,
                CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS, cudss_config, cudss_solver_data, batch_A, nullptr, nullptr));
            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION, cudss_config, cudss_solver_data, batch_A, nullptr, nullptr));
        }

        CHECK_CUDA(cudaDeviceSynchronize());
        SPDLOG_INFO("[{}] [factorize_submatrix] [{}] [n_sparse={}] [n_sparse_dofs={}]",
                     name(), elapsed_seconds(phase_begin), sparse_batch_count, total_sparse_dofs);
    }

    void GPUHybridSolver::pcg_solve(thrust::device_vector<double> &rhs, thrust::device_vector<double> &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond)
    {
        double bi_prod, abs_eps, rel_eps, gamma, old_gamma;

        thrust::device_vector<double> r(rhs.size());
        thrust::device_vector<double> p(rhs.size());
        thrust::device_vector<double> z(rhs.size());
        thrust::device_vector<double> z2(rhs.size());
        thrust::device_vector<double> buffer(rhs.size());

        num_iterations = 0;
        stop_reason = "max_iter";
        const auto krylov_begin = clock::now();

        {
            auto phase_begin = clock::now();

            bi_prod = dot(rhs, rhs);

            if (bi_prod > 0.0)
            {
                rel_eps = rel_conv_tol_ * rel_conv_tol_;
                abs_eps = abs_conv_tol_ * abs_conv_tol_;
            }
            else
            {
                thrust::fill(result.begin(), result.end(), 0.0);
                num_iterations = 0;
                final_res_norm = 0;
                stop_reason = "zero_rhs";
                return;
            }

            matmul(result, buffer);

            vector_copy(rhs, r);
            vector_add(-1.0, buffer, r);
            record_iteration(std::sqrt(dot(r, r) / bi_prod), krylov_begin);

            apply_precond(precond, r, z, buffer, z2);

            vector_copy(z, p);

            gamma = dot(r, z);
            old_gamma = gamma;
            SPDLOG_INFO("[{}] [pre_loop] [{:.6f}] [rhs_norm={}]", name(), elapsed_seconds(phase_begin), sqrt(bi_prod));
        }

        for (int k = 0; k < max_iter_; ++k)
        {
            auto phase_begin = clock::now();
            num_iterations = k + 1;

            matmul(p, buffer);
            double sdotp = dot(p, buffer);

            if (!std::isfinite(sdotp) || !std::isfinite(gamma))
            {
                stop_reason = "nan";
                break;
            }

            if (sdotp == 0.0)
            {
                SPDLOG_INFO("[{}] [err_zero_sdotp] [0.000000]", name());
                stop_reason = "zero_sdotp";
                break;
            }

            double alpha = gamma / sdotp;

            if (alpha <= 0.0)
            {
                SPDLOG_INFO("[{}] [err_negative_alpha] [0.000000]", name());
                stop_reason = "negative_alpha";
                break;
            }
            else if (alpha < __DBL_MIN__)
            {
                SPDLOG_INFO("[{}] [err_subnormal_alpha] [0.000000]", name());
                stop_reason = "subnormal_alpha";
                break;
            }

            vector_add(alpha, p, result);
            vector_add(-1.0 * alpha, buffer, r);
            double i_prod = dot(r, r);
            record_iteration(std::sqrt(i_prod / bi_prod), krylov_begin);

            if (!std::isfinite(i_prod))
            {
                stop_reason = "nan";
                break;
            }

            if (rel_eps > 0 && (i_prod / bi_prod) < rel_eps)
            {
                SPDLOG_INFO("[{}] [converged_rel] [0.000000]", name());
                stop_reason = "converged";
                break;
            }

            if (abs_eps > 0 && i_prod < abs_eps)
            {
                SPDLOG_INFO("[{}] [converged_abs] [0.000000]", name());
                stop_reason = "converged";
                break;
            }

            thrust::fill(z.begin(), z.end(), 0.0);
            apply_precond(precond, r, z, buffer, z2);

            gamma = dot(r, z);

            double beta = gamma / old_gamma;
            old_gamma = gamma;

            vector_scale(beta, p);
            vector_add(1.0, z, p);

            CHECK_CUDA(cudaDeviceSynchronize());
            GPUHYBRID_LOG_ITER("[{}] [pcg_iter] [{:.6f}] [iter={}] [residual={}]", name(), elapsed_seconds(phase_begin), k, sqrt(i_prod));
        }
    }

    void GPUHybridSolver::record_iteration(double rel_res, const std::chrono::steady_clock::time_point &begin)
    {
        if (record_residual_history)
        {
            residual_history.push_back(rel_res);
            time_history.push_back(elapsed_seconds(begin));
        }
    }

    void GPUHybridSolver::apply_precond(const HYPRE_Solver &precond, thrust::device_vector<double> &r, thrust::device_vector<double> &z, thrust::device_vector<double> &buffer, thrust::device_vector<double> &z2)
    {
        switch (preconditioner_type)
        {
        case PreconditionerType::HYBRID:
            custom_mixed_precond_iter(precond, r, z, buffer, z2);
            break;
        case PreconditionerType::JACOBI:
            thrust::transform(thrust::device, r.begin(), r.end(), d_inv_diag.begin(), z.begin(), thrust::multiplies<double>());
            break;
        case PreconditionerType::IC:
            ic_precond_iter(r, z);
            break;
        case PreconditionerType::ILU:
            thrust::fill(z.begin(), z.end(), 0.0);
            ilu_precond_iter(precond, r, z);
            break;
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // HYPRE_ILUSolve is a Richardson step; with x = 0 on entry it returns
    // M^{-1} b (plus one matvec for the initial residual, as in hypre's own
    // Krylov solvers).
    void GPUHybridSolver::ilu_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double> &b, thrust::device_vector<double> &x)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_b;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_b, b);

        HYPRE_ILUSolve(precond, parcsr_A, par_b, par_x);
    }

    void GPUHybridSolver::setup_jacobi(const StiffnessMatrix &At)
    {
        // Column k of At is row k of A.
        const int n = At.rows();
        std::vector<double> inv_diag(n, 1.0);
        int zero_diag = 0;
        for (int k = 0; k < n; ++k)
        {
            double d = 0.0;
            for (StiffnessMatrix::InnerIterator it(At, k); it; ++it)
            {
                if (it.row() == k)
                {
                    d = it.value();
                    break;
                }
            }
            if (d == 0.0)
                ++zero_diag; // left unscaled (saddle-point blocks)
            else
                inv_diag[k] = 1.0 / d;
        }
        d_inv_diag = inv_diag;
        info_["jacobi_zero_diagonals"] = zero_diag;
    }

    // IC(0) of A's lower triangle with cuSPARSE csric02. A pivot that breaks
    // down (zero, negative or not finite) restarts the factorization with the
    // diagonal shifted by shift * |a_ii|, shift = ic_initial_shift doubling up
    // to 10 times -- the same strategy as Eigen's IncompleteCholesky on the CPU.
    void GPUHybridSolver::setup_ic(const StiffnessMatrix &At)
    {
        free_ic_memory();
        const int n = At.rows();

        // Lower triangle in CSR: row k of A is column k of At; keep j <= k.
        std::vector<int> outer(n + 1, 0), inner, diag_pos(n, -1);
        std::vector<double> values;
        for (int k = 0; k < n; ++k)
        {
            outer[k] = inner.size();
            for (StiffnessMatrix::InnerIterator it(At, k); it; ++it)
            {
                const int j = it.row();
                if (j > k)
                    continue;
                if (j == k)
                    diag_pos[k] = inner.size();
                inner.push_back(j);
                values.push_back(it.value());
            }
        }
        outer[n] = inner.size();
        const int nnz = inner.size();
        for (int k = 0; k < n; ++k)
        {
            if (diag_pos[k] < 0)
                throw std::runtime_error("[GPUHybrid] IC: row " + std::to_string(k) + " has no diagonal entry");
        }

        d_ic_outer = outer;
        d_ic_inner = inner;
        d_ic_values.resize(nnz);
        int *raw_outer = thrust::raw_pointer_cast(d_ic_outer.data());
        int *raw_inner = thrust::raw_pointer_cast(d_ic_inner.data());
        double *raw_values = thrust::raw_pointer_cast(d_ic_values.data());

        cusparseMatDescr_t descr;
        CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
        CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
        CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        csric02Info_t ic_info;
        CHECK_CUSPARSE(cusparseCreateCsric02Info(&ic_info));

        thrust::copy(values.begin(), values.end(), d_ic_values.begin());
        int buffer_size = 0;
        CHECK_CUSPARSE(cusparseDcsric02_bufferSize(cusparse_handle, n, nnz, descr, raw_values, raw_outer, raw_inner, ic_info, &buffer_size));
        thrust::device_vector<char> buffer(std::max(buffer_size, 1));
        void *raw_buffer = thrust::raw_pointer_cast(buffer.data());

        double shift = 0.0;
        bool ok = false;
        for (int attempt = 0; attempt <= 10 && !ok; ++attempt)
        {
            std::vector<double> shifted = values;
            for (int k = 0; k < n; ++k)
                shifted[diag_pos[k]] += shift * std::abs(values[diag_pos[k]]);
            thrust::copy(shifted.begin(), shifted.end(), d_ic_values.begin());

            CHECK_CUSPARSE(cusparseDcsric02_analysis(cusparse_handle, n, nnz, descr, raw_values, raw_outer, raw_inner, ic_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, raw_buffer));
            int position = -1;
            if (cusparseXcsric02_zeroPivot(cusparse_handle, ic_info, &position) == CUSPARSE_STATUS_ZERO_PIVOT)
                break; // structural zero: no shift helps

            CHECK_CUSPARSE(cusparseDcsric02(cusparse_handle, n, nnz, descr, raw_values, raw_outer, raw_inner, ic_info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, raw_buffer));
            const bool zero_pivot = cusparseXcsric02_zeroPivot(cusparse_handle, ic_info, &position) == CUSPARSE_STATUS_ZERO_PIVOT;
            CHECK_CUDA(cudaDeviceSynchronize());

            // A negative pivot shows up as a NaN on L's diagonal.
            const int bad = thrust::count_if(thrust::device, d_ic_values.begin(), d_ic_values.end(),
                                             [] __device__(double v) { return !isfinite(v); });
            ok = !zero_pivot && bad == 0;
            if (!ok)
                shift = shift == 0.0 ? ic_initial_shift : 2.0 * shift;
        }

        CHECK_CUSPARSE(cusparseDestroyCsric02Info(ic_info));
        CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
        info_["ic_shift"] = shift;
        if (!ok)
        {
            free_ic_memory();
            throw std::runtime_error("[GPUHybrid] ic_factorization_failed");
        }

        // Triangular-solve plans: L t = r, then L^T z = t.
        CHECK_CUSPARSE(cusparseCreateCsr(&ic_L, n, n, nnz, raw_outer, raw_inner, raw_values,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
        cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
        CHECK_CUSPARSE(cusparseSpMatSetAttribute(ic_L, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill)));
        CHECK_CUSPARSE(cusparseSpMatSetAttribute(ic_L, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

        d_ic_tmp.resize(n);
        thrust::device_vector<double> tmp2(n);
        cusparseDnVecDescr_t vx, vy;
        CHECK_CUSPARSE(cusparseCreateDnVec(&vx, n, thrust::raw_pointer_cast(d_ic_tmp.data()), CUDA_R_64F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vy, n, thrust::raw_pointer_cast(tmp2.data()), CUDA_R_64F));
        const double one = 1.0;
        size_t size_L = 0, size_Lt = 0;
        CHECK_CUSPARSE(cusparseSpSV_createDescr(&ic_spsv_L));
        CHECK_CUSPARSE(cusparseSpSV_createDescr(&ic_spsv_Lt));
        CHECK_CUSPARSE(cusparseSpSV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, ic_L, vx, vy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, ic_spsv_L, &size_L));
        CHECK_CUSPARSE(cusparseSpSV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, ic_L, vx, vy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, ic_spsv_Lt, &size_Lt));
        d_ic_buffer_L.resize(std::max<size_t>(size_L, 1));
        d_ic_buffer_Lt.resize(std::max<size_t>(size_Lt, 1));
        CHECK_CUSPARSE(cusparseSpSV_analysis(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, ic_L, vx, vy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, ic_spsv_L, thrust::raw_pointer_cast(d_ic_buffer_L.data())));
        CHECK_CUSPARSE(cusparseSpSV_analysis(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, ic_L, vx, vy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, ic_spsv_Lt, thrust::raw_pointer_cast(d_ic_buffer_Lt.data())));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vx));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vy));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    void GPUHybridSolver::ic_precond_iter(thrust::device_vector<double> &r, thrust::device_vector<double> &z)
    {
        const int n = r.size();
        const double one = 1.0;
        cusparseDnVecDescr_t vr, vt, vz;
        CHECK_CUSPARSE(cusparseCreateDnVec(&vr, n, thrust::raw_pointer_cast(r.data()), CUDA_R_64F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vt, n, thrust::raw_pointer_cast(d_ic_tmp.data()), CUDA_R_64F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vz, n, thrust::raw_pointer_cast(z.data()), CUDA_R_64F));
        CHECK_CUSPARSE(cusparseSpSV_solve(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, ic_L, vr, vt, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, ic_spsv_L));
        CHECK_CUSPARSE(cusparseSpSV_solve(cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, &one, ic_L, vt, vz, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, ic_spsv_Lt));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vr));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vt));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vz));
    }

    void GPUHybridSolver::free_ic_memory()
    {
        if (ic_spsv_L)
        {
            cusparseSpSV_destroyDescr(ic_spsv_L);
            ic_spsv_L = nullptr;
        }
        if (ic_spsv_Lt)
        {
            cusparseSpSV_destroyDescr(ic_spsv_Lt);
            ic_spsv_Lt = nullptr;
        }
        if (ic_L)
        {
            cusparseDestroySpMat(ic_L);
            ic_L = nullptr;
        }
        d_ic_outer.clear();
        d_ic_inner.clear();
        d_ic_values.clear();
        d_ic_tmp.clear();
        d_ic_buffer_L.clear();
        d_ic_buffer_Lt.clear();
    }

    // Whole-matrix cuDSS factorization (direct mode). At's arrays are A in CSR.
    void GPUHybridSolver::factorize_direct(const StiffnessMatrix &At)
    {
        auto phase_begin = clock::now();
        free_direct_memory();

        const int n = At.rows();
        const int nnz = At.nonZeros();
        d_direct_outer.assign(At.outerIndexPtr(), At.outerIndexPtr() + n + 1);
        d_direct_inner.assign(At.innerIndexPtr(), At.innerIndexPtr() + nnz);
        d_direct_values.assign(At.valuePtr(), At.valuePtr() + nnz);

        const bool lu = direct_factorization == DirectFactorization::LU;
        CHECK_CUDSS(cudssConfigCreate(&direct_config));
        CHECK_CUDSS(cudssDataCreate(cudss_handle, &direct_data));
        CHECK_CUDSS(cudssMatrixCreateCsr(
            &direct_A, n, n, nnz,
            thrust::raw_pointer_cast(d_direct_outer.data()), nullptr,
            thrust::raw_pointer_cast(d_direct_inner.data()),
            thrust::raw_pointer_cast(d_direct_values.data()),
            CUDA_R_32I, CUDA_R_64F, lu ? CUDSS_MTYPE_GENERAL : CUDSS_MTYPE_SPD,
            CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

        CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS, direct_config, direct_data, direct_A, nullptr, nullptr));
        CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION, direct_config, direct_data, direct_A, nullptr, nullptr));
        CHECK_CUDA(cudaDeviceSynchronize());

        // info != 0: zero pivot (LU) or a matrix that is not positive definite (LLT)
        int status = 0;
        size_t written = 0;
        CHECK_CUDSS(cudssDataGet(cudss_handle, direct_data, CUDSS_DATA_INFO, &status, sizeof(status), &written));
        int64_t factor_nnz = 0;
        CHECK_CUDSS(cudssDataGet(cudss_handle, direct_data, CUDSS_DATA_LU_NNZ, &factor_nnz, sizeof(factor_nnz), &written));

        info_["direct_backend"] = lu ? "cuDSS_LU" : "cuDSS_LLT";
        info_["factor_nnz"] = factor_nnz;
        info_["direct_factorize_time"] = elapsed_seconds(phase_begin);
        SPDLOG_INFO("[{}] [direct_factorize] [{:.6f}]", name(), elapsed_seconds(phase_begin));
        if (status != 0)
        {
            free_direct_memory();
            throw std::runtime_error("[GPUHybrid] cuDSS factorization failed (info " + std::to_string(status) + ")");
        }
    }

    void GPUHybridSolver::solve_direct(thrust::device_vector<double> &b, thrust::device_vector<double> &x)
    {
        if (!direct_A)
            throw std::runtime_error("[GPUHybrid] solve called before factorize");
        const int n = b.size();
        cudssMatrix_t mb, mx;
        CHECK_CUDSS(cudssMatrixCreateDn(&mb, n, 1, n, thrust::raw_pointer_cast(b.data()), CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        CHECK_CUDSS(cudssMatrixCreateDn(&mx, n, 1, n, thrust::raw_pointer_cast(x.data()), CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_SOLVE, direct_config, direct_data, direct_A, mx, mb));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDSS(cudssMatrixDestroy(mb));
        CHECK_CUDSS(cudssMatrixDestroy(mx));
    }

    void GPUHybridSolver::free_direct_memory()
    {
        if (direct_A)
        {
            CHECK_CUDSS(cudssMatrixDestroy(direct_A));
            direct_A = nullptr;
        }
        if (direct_data)
        {
            CHECK_CUDSS(cudssDataDestroy(cudss_handle, direct_data));
            direct_data = nullptr;
        }
        if (direct_config)
        {
            CHECK_CUDSS(cudssConfigDestroy(direct_config));
            direct_config = nullptr;
        }
        d_direct_outer.clear();
        d_direct_inner.clear();
        d_direct_values.clear();
    }

    // Restarted, right-preconditioned GMRES(m), as in CPUHybridSolver::gmres_solve:
    // x = x0 + M^{-1} V y, CGS2 orthogonalization, each cycle starting from the
    // recomputed true residual. The basis V (n x (m+1), column-major) lives on
    // the device and is projected with cuBLAS gemv; the small Hessenberg
    // least-squares problem is solved on the host.
    void GPUHybridSolver::gmres_solve(thrust::device_vector<double> &rhs, thrust::device_vector<double> &result, HYPRE_Solver &precond)
    {
        num_iterations = 0;
        stop_reason = "max_iter";
        const auto krylov_begin = clock::now();

        const int n = rhs.size();
        const int m = gmres_restart;

        const double b_norm = std::sqrt(dot(rhs, rhs));
        if (!(b_norm > 0.0))
        {
            thrust::fill(result.begin(), result.end(), 0.0);
            stop_reason = b_norm == 0.0 ? "zero_rhs" : "nan";
            return;
        }
        const double target = std::max(rel_conv_tol_ * b_norm, abs_conv_tol_);

        thrust::device_vector<double> V((size_t)n * (m + 1));
        thrust::device_vector<double> res(n), v(n), z(n), w(n), buffer(n), z2(n), d_h(m + 1), d_y(m);
        double *raw_V = thrust::raw_pointer_cast(V.data());
        double *raw_w = thrust::raw_pointer_cast(w.data());
        double *raw_h = thrust::raw_pointer_cast(d_h.data());

        Eigen::MatrixXd H(m + 1, m);
        Eigen::VectorXd cs(m), sn(m), g(m + 1), h(m + 1);
        const double one = 1.0, zero = 0.0, minus_one = -1.0;

        bool first_cycle = true;
        while (true)
        {
            matmul(result, w);
            vector_copy(rhs, res);
            vector_add(-1.0, w, res);
            const double beta = std::sqrt(dot(res, res));
            if (first_cycle)
            {
                record_iteration(beta / b_norm, krylov_begin);
                first_cycle = false;
            }
            if (!std::isfinite(beta))
            {
                stop_reason = "nan";
                break;
            }
            if (beta <= target)
            {
                stop_reason = "converged";
                break;
            }
            if (num_iterations >= max_iter_)
            {
                break;
            }

            thrust::transform(thrust::device, res.begin(), res.end(), V.begin(), [beta] __device__(double a) { return a / beta; });
            H.setZero();
            g.setZero();
            g(0) = beta;

            int k = 0;
            bool broke_down = false;
            for (int j = 0; j < m && num_iterations < max_iter_; ++j)
            {
                auto phase_begin = clock::now();

                // w = A M^{-1} v_j
                thrust::copy(thrust::device, V.begin() + (size_t)j * n, V.begin() + (size_t)(j + 1) * n, v.begin());
                thrust::fill(z.begin(), z.end(), 0.0);
                CHECK_CUDA(cudaDeviceSynchronize());
                apply_precond(precond, v, z, buffer, z2);
                matmul(z, w);

                for (int pass = 0; pass < 2; ++pass)
                {
                    CHECK_CUBLAS(cublasDgemv(cublas_handle, CUBLAS_OP_T, n, j + 1, &one, raw_V, n, raw_w, 1, &zero, raw_h, 1));
                    CHECK_CUBLAS(cublasDgemv(cublas_handle, CUBLAS_OP_N, n, j + 1, &minus_one, raw_V, n, raw_h, 1, &one, raw_w, 1));
                    CHECK_CUDA(cudaDeviceSynchronize());
                    thrust::copy(d_h.begin(), d_h.begin() + j + 1, h.data());
                    H.col(j).head(j + 1) += h.head(j + 1);
                }
                const double h_next = std::sqrt(dot(w, w));
                H(j + 1, j) = h_next;

                for (int i = 0; i < j; ++i)
                {
                    const double t = cs(i) * H(i, j) + sn(i) * H(i + 1, j);
                    H(i + 1, j) = -sn(i) * H(i, j) + cs(i) * H(i + 1, j);
                    H(i, j) = t;
                }
                const double rho = std::hypot(H(j, j), H(j + 1, j));
                cs(j) = rho > 0.0 ? H(j, j) / rho : 1.0;
                sn(j) = rho > 0.0 ? H(j + 1, j) / rho : 0.0;
                H(j, j) = rho;
                H(j + 1, j) = 0.0;
                g(j + 1) = -sn(j) * g(j);
                g(j) = cs(j) * g(j);

                ++num_iterations;
                k = j + 1;
                const double res_estimate = std::abs(g(j + 1));
                record_iteration(res_estimate / b_norm, krylov_begin);
                GPUHYBRID_LOG_ITER("[{}] [gmres_iter] [{:.6f}] [iter={}] [residual={}]", name(), elapsed_seconds(phase_begin), num_iterations, res_estimate);

                if (!std::isfinite(res_estimate) || !std::isfinite(h_next))
                {
                    broke_down = true;
                    break;
                }
                if (res_estimate <= target)
                {
                    break; // confirmed by the true residual at the top of the loop
                }
                if (h_next == 0.0)
                {
                    break; // invariant subspace: the update below is exact
                }
                thrust::transform(thrust::device, w.begin(), w.end(), V.begin() + (size_t)(j + 1) * n, [h_next] __device__(double a) { return a / h_next; });
            }

            if (broke_down)
            {
                stop_reason = "nan";
                break;
            }

            // x += M^{-1} V_k y, with R_k y = g_k
            const Eigen::VectorXd y = H.topLeftCorner(k, k).triangularView<Eigen::Upper>().solve(g.head(k));
            thrust::copy(y.data(), y.data() + k, d_y.begin());
            CHECK_CUBLAS(cublasDgemv(cublas_handle, CUBLAS_OP_N, n, k, &one, raw_V, n, thrust::raw_pointer_cast(d_y.data()), 1, &zero, thrust::raw_pointer_cast(v.data()), 1));
            thrust::fill(z.begin(), z.end(), 0.0);
            CHECK_CUDA(cudaDeviceSynchronize());
            apply_precond(precond, v, z, buffer, z2);
            vector_add(1.0, z, result);
        }
    }

    GPUHybridSolver::~GPUHybridSolver()
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
            A = nullptr;
        }

        free_device_memory();
        free_contact_patch_dense_memory();
        free_direct_memory();
        free_ic_memory();

        if (cusparse_handle)
        {
            cusparseDestroy(cusparse_handle);
            cusparse_handle = nullptr;
        }
        if (cudss_handle)
        {
            cudssDestroy(cudss_handle);
            cudss_handle = nullptr;
        }
        if (cublas_handle)
        {
            cublasDestroy(cublas_handle);
            cublas_handle = nullptr;
        }
    }

    void GPUHybridSolver::free_contact_patch_dense_memory()
    {
        num_contact_patches_ = 0;
        total_contact_patch_real_dofs_ = 0;
        num_unique_overlap_dofs_ = 0;

        d_dense_A.clear();
        d_dense_A_ptrs.clear();
        d_dense_pivot.clear();
        d_dense_info.clear();
        d_dense_b.clear();
        d_dense_b_ptrs.clear();
        d_contact_dof_map.clear();
        d_contact_real_to_dense_slot.clear();
        d_reduce_perm.clear();
        d_reduce_keys.clear();
        d_reduce_values.clear();
        d_reduce_unique_keys.clear();
        d_reduce_unique_values.clear();
    }

    void GPUHybridSolver::free_device_memory()
    {
        // Destroy cuDSS Opaque Structures
        if (batch_A)
        {
            CHECK_CUDSS(cudssMatrixDestroy(batch_A));
            batch_A = nullptr;
        }
        if (batch_x)
        {
            CHECK_CUDSS(cudssMatrixDestroy(batch_x));
            batch_x = nullptr;
        }
        if (batch_b)
        {
            CHECK_CUDSS(cudssMatrixDestroy(batch_b));
            batch_b = nullptr;
        }
        if (cudss_solver_data)
        {
            CHECK_CUDSS(cudssDataDestroy(cudss_handle, cudss_solver_data));
            cudss_solver_data = nullptr;
        }
        if (cudss_config)
        {
            CHECK_CUDSS(cudssConfigDestroy(cudss_config));
            cudss_config = nullptr;
        }
    }
} // namespace polysolve::linear