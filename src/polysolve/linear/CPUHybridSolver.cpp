
////////////////////////////////////////////////////////////////////////////////
#include "CPUHybridSolver.hpp"
////////////////////////////////////////////////////////////////////////////////

#include "hybrid_utils/DisjointSet.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>

#include <HYPRE_utilities.h>
#include <Eigen/SparseCholesky>

#if POLYSOLVE_WITH_ACCELERATE
#include <Eigen/AccelerateSupport>
#endif

#if POLYSOLVE_WITH_MKL
#include <Eigen/PardisoSupport>
#endif

#ifdef POLYSOLVE_WITH_STRUMPACK
#include <StrumpackSparseSolver.hpp>
#endif

#include <chrono>
#include <spdlog/spdlog.h>
#include <thread>

#include "_hypre_IJ_mv.h"
#include "_hypre_parcsr_mv.h"

namespace polysolve::linear
{

    ////////////////////////////////////////////////////////////////////////////////
    namespace
    {
        using clock = std::chrono::steady_clock;

        double elapsed_seconds(const std::chrono::time_point<clock> &begin)
        {
            return std::chrono::duration<double>(clock::now() - begin).count();
        }

#ifdef POLYSOLVE_WITH_STRUMPACK
        // A subdomain factorized by STRUMPACK, under use_strumpack. Only the
        // rank that owns the subdomain -- a thread of this process -- ever
        // touches it, and STRUMPACK is built without MPI or OpenMP, so it runs
        // sequentially on that thread.
        //
        // Note that STRUMPACK's solver installs a process-wide new-handler
        // for its lifetime, which aborts on out-of-memory rather than
        // throwing std::bad_alloc.
        class StrumpackWrapper : public AbstractSolver
        {
            strumpack::StrumpackSparseSolver<double, int> solver{/*verbose=*/false};

        public:
            StrumpackWrapper(CPUHybridSolver::StrumpackCompression compression, double compression_rel_tol)
            {
                auto &opts = solver.options();

                // Apply the factorization exactly once per solve. The default
                // (AUTO) wraps it in iterative refinement -- or, once
                // compressed, in GMRES -- which would make this preconditioner
                // nonlinear under PCG.
                opts.set_Krylov_solver(strumpack::KrylovSolver::DIRECT);
                // Subdomains are principal submatrices of an SPD system, so
                // their diagonal needs no help from MC64's row permutation,
                // which would only throw away their symmetry.
                opts.set_matching(strumpack::MatchingJob::NONE);

                switch (compression)
                {
                case CPUHybridSolver::StrumpackCompression::NONE:
                    opts.set_compression(strumpack::CompressionType::NONE);
                    break;
                case CPUHybridSolver::StrumpackCompression::BLR:
                    opts.set_compression(strumpack::CompressionType::BLR);
                    break;
                case CPUHybridSolver::StrumpackCompression::HSS:
                    opts.set_compression(strumpack::CompressionType::HSS);
                    break;
                }
                opts.set_compression_rel_tol(compression_rel_tol);
            }

            void compute(const Eigen::SparseMatrix<double> &A) override
            {
                // STRUMPACK takes CSR, and keeps a copy of its own.
                Eigen::SparseMatrix<double, Eigen::RowMajor> A_csr = A;
                A_csr.makeCompressed();

                // Don't promise a symmetric pattern: an assembled matrix can
                // store an explicit zero on one side of the diagonal only,
                // and STRUMPACK would trust the promise.
                solver.set_csr_matrix(A_csr.rows(), A_csr.outerIndexPtr(), A_csr.innerIndexPtr(), A_csr.valuePtr(), /*symmetric_pattern=*/false);

                // The reordering is METIS nested dissection, and METIS keeps
                // its random number generator in a process-wide static that
                // the other ranks' subdomains, reordered at the same time,
                // would race on. So reorder one subdomain at a time; METIS
                // reseeds on every call, which also makes the orderings
                // deterministic. The numeric factorization stays concurrent.
                strumpack::ReturnCode status;
                {
                    static std::mutex metis_mutex;
                    std::lock_guard<std::mutex> lock(metis_mutex);
                    status = solver.reorder();
                }
                if (status == strumpack::ReturnCode::SUCCESS)
                {
                    status = solver.factor();
                }
                if (status != strumpack::ReturnCode::SUCCESS)
                {
                    SPDLOG_ERROR("[CPUHybrid] STRUMPACK failed to factorize a subdomain of size {} (return code {})", A.rows(), static_cast<int>(status));
                }
            }

            Eigen::VectorXd solve(const Eigen::VectorXd &b) override
            {
                Eigen::VectorXd x(b.size());
                solver.solve(b.data(), x.data());
                return x;
            }
        };
#endif
    } // namespace

    // Workers (nanompi ranks 1..N-1, each a thread of this process) run the
    // very same collective code paths as rank 0 -- see run_worker_loop --
    // so an unguarded SPDLOG_INFO below would print once per rank. Route
    // rank-scoped logging through this macro instead of muting the logger
    // globally: spdlog::set_level() mutates the single process-wide default
    // logger, so calling it from a worker thread silences rank 0 as well.
#define CPUHYBRID_LOG_INFO(...)      \
    do                               \
    {                                \
        if (myid == 0)               \
            SPDLOG_INFO(__VA_ARGS__); \
    } while (0)

    // Per-PCG-iteration timing logs (matmul, amg_v_cycle, subdomain_solve,
    // pcg_iter) go through this instead of CPUHYBRID_LOG_INFO, so they can be
    // silenced with detailed_log=false without affecting any other log line.
#define CPUHYBRID_LOG_ITER(...)                   \
    do                                             \
    {                                              \
        if (myid == 0 && detailed_log)             \
            SPDLOG_INFO(__VA_ARGS__);               \
    } while (0)

    CPUHybridSolver::CPUHybridSolver()
    {
        // Ranks are threads of this process. The first solver constructed
        // spawns them; the calling thread becomes rank 0 and drives.
        ensure_ranks();

        if (!HYPRE_Initialized())
        {
            HYPRE_Initialize();
        }

        // get MPI rank information
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        if (is_running_worker_loop)
        {
            return;
        }

        Eigen::setNbThreads(1);
        HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
        HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);

        if (myid == 0)
        {
            spdlog::flush_on(spdlog::level::info);
        }

        // A worker rank never reaches here: it is already inside
        // run_worker_loop() on its own thread (see ensure_ranks below).

        solver_id = next_id++;

        SolverCmd cmd = CMD_CREATE;
        MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&solver_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Set solver parameters
    void CPUHybridSolver::set_parameters(const json &params)
    {
        if (myid == 0)
        {
            SolverCmd cmd = CMD_SET_PARAMETERS;
            MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&solver_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

        json shared_params;
        std::string json_str;
        int str_size = 0;

        if (myid == 0)
        {
            shared_params = params; // Use the incoming params on root
            json_str = shared_params.dump();
            str_size = static_cast<int>(json_str.size());
        }

        MPI_Bcast(&str_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (myid != 0)
        {
            json_str.resize(str_size);
        }

        MPI_Bcast(json_str.data(), str_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        shared_params = json::parse(json_str);

        if (shared_params.contains("CPUHybrid"))
        {
            if (shared_params["CPUHybrid"].contains("max_iter"))
            {
                max_iter_ = shared_params["CPUHybrid"]["max_iter"];
            }
            if (shared_params["CPUHybrid"].contains("relative_tolerance"))
            {
                rel_conv_tol_ = shared_params["CPUHybrid"]["relative_tolerance"];
            }
            if (shared_params["CPUHybrid"].contains("absolute_tolerance"))
            {
                abs_conv_tol_ = shared_params["CPUHybrid"]["absolute_tolerance"];
            }
            if (shared_params["CPUHybrid"].contains("theta"))
            {
                theta = shared_params["CPUHybrid"]["theta"];
            }
            if (shared_params["CPUHybrid"].contains("decompose_subdomains"))
            {
                decompose_subdomains = shared_params["CPUHybrid"]["decompose_subdomains"];
            }
            if (shared_params["CPUHybrid"].contains("min_subdomain_size"))
            {
                min_subdomain_size = shared_params["CPUHybrid"]["min_subdomain_size"];
            }
            if (shared_params["CPUHybrid"].contains("max_subdomain_size"))
            {
                max_subdomain_size = shared_params["CPUHybrid"]["max_subdomain_size"];
            }
            if (shared_params["CPUHybrid"].contains("expand_subdomains"))
            {
                expand_subdomains = shared_params["CPUHybrid"]["expand_subdomains"];
            }
            if (shared_params["CPUHybrid"].contains("gmm_jump_threshold"))
            {
                gmm_jump_threshold = shared_params["CPUHybrid"]["gmm_jump_threshold"];
            }
            if (shared_params["CPUHybrid"].contains("gmm_tol"))
            {
                gmm_tol = shared_params["CPUHybrid"]["gmm_tol"];
            }
            if (shared_params["CPUHybrid"].contains("max_gmm_iterations"))
            {
                max_gmm_iterations = shared_params["CPUHybrid"]["max_gmm_iterations"];
            }
            if (shared_params["CPUHybrid"].contains("conditioning_threshold"))
            {
                conditioning_threshold = shared_params["CPUHybrid"]["conditioning_threshold"];
            }
            if (shared_params["CPUHybrid"].contains("additive_mode"))
            {
                additive_mode = shared_params["CPUHybrid"]["additive_mode"];
            }
            if (shared_params["CPUHybrid"].contains("select_bad_dofs_from_l1_norm"))
            {
                select_bad_dofs_from_l1_row_norm = shared_params["CPUHybrid"]["select_bad_dofs_from_l1_norm"];
            }
            if (shared_params["CPUHybrid"].contains("subdomain_selection_strategy"))
            {
                const std::string strategy_str = shared_params["CPUHybrid"]["subdomain_selection_strategy"];
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
            if (shared_params["CPUHybrid"].contains("contact_patch_schwarz"))
            {
                contact_patch_schwarz = shared_params["CPUHybrid"]["contact_patch_schwarz"];
            }
            if (shared_params["CPUHybrid"].contains("contact_patch_remove_covered"))
            {
                contact_patch_remove_covered = shared_params["CPUHybrid"]["contact_patch_remove_covered"];
            }
            if (shared_params["CPUHybrid"].contains("ras_correction"))
            {
                ras_correction = shared_params["CPUHybrid"]["ras_correction"];
            }
            if (shared_params["CPUHybrid"].contains("use_strumpack"))
            {
                use_strumpack = shared_params["CPUHybrid"]["use_strumpack"];
            }
            if (shared_params["CPUHybrid"].contains("strumpack_compression"))
            {
                const std::string compression_str = shared_params["CPUHybrid"]["strumpack_compression"];
                if (compression_str == "none")
                {
                    strumpack_compression = StrumpackCompression::NONE;
                }
                else if (compression_str == "blr")
                {
                    strumpack_compression = StrumpackCompression::BLR;
                }
                else if (compression_str == "hss")
                {
                    strumpack_compression = StrumpackCompression::HSS;
                }
                else
                {
                    throw std::runtime_error("Unknown STRUMPACK compression: " + compression_str);
                }
            }
            if (shared_params["CPUHybrid"].contains("strumpack_compression_rel_tol"))
            {
                strumpack_compression_rel_tol = shared_params["CPUHybrid"]["strumpack_compression_rel_tol"];
            }
            if (shared_params["CPUHybrid"].contains("detailed_log"))
            {
                detailed_log = shared_params["CPUHybrid"]["detailed_log"];
            }
        }
    }

    // Set block size for multigrid solvers
    void CPUHybridSolver::set_block_size(int block_size)
    {
        if (myid == 0)
        {
            SolverCmd cmd = CMD_SET_BLOCK_SIZE;
            MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&solver_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

        // Broadcast the block size itself, so worker ranks (which pass in a
        // dummy value from run_worker_loop) end up with the root's value.
        MPI_Bcast(&block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        dimension_ = block_size;
    }

    // Set the function (block) assigned to each row for multigrid solvers
    void CPUHybridSolver::set_block_mapping(const Eigen::VectorXi &block_mapping)
    {
        if (myid == 0)
        {
            SolverCmd cmd = CMD_SET_BLOCK_MAPPING;
            MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&solver_id, 1, MPI_INT, 0, MPI_COMM_WORLD);

            block_mapping_ = block_mapping;
        }

        // Broadcast the full (global) mapping to every rank; each rank keeps the
        // whole thing since the local row range ([starts[myid], ends[myid]]) is
        // only known once factorize() has partitioned the rows.
        int size = myid == 0 ? (int)block_mapping_.size() : 0;
        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (myid != 0)
            block_mapping_.resize(size);

        if (size > 0)
            MPI_Bcast(block_mapping_.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    }

    void CPUHybridSolver::check_settings() const
    {
        if (myid != 0)
        {
            return;
        }

        if (subdomain_selection_strategy == SubdomainSelectionStrategy::APOSTERIORI)
        {
            throw std::runtime_error("A Posteriori subdomain selection strategy is not yet implemented!");
        }

        if (contact_patch_schwarz && ras_correction)
        {
            throw std::runtime_error("contact_patch_schwarz and ras_correction cannot both be enabled!");
        }

#ifndef POLYSOLVE_WITH_STRUMPACK
        if (use_strumpack)
        {
            throw std::runtime_error("use_strumpack requires polysolve to be built with POLYSOLVE_WITH_STRUMPACK=ON!");
        }
#endif
    }

    void CPUHybridSolver::get_info(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void CPUHybridSolver::factorize(const StiffnessMatrix &Ain)
    {
        if (myid == 0)
        {
            SolverCmd cmd = CMD_FACTORIZE;
            MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&solver_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

        check_settings();

        int rows, cols, nnz;
        if (myid == 0)
        {
            CPUHYBRID_LOG_INFO("[{}] [start_solve] [0.000000] [num_procs={}] [problem_size={}]", name(), num_procs, Ain.rows());
            rows = Ain.rows();
            cols = Ain.cols();
            nnz = Ain.nonZeros();
        }

        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

        partition_ranks(rows);

        MPI_Win A_win;
        double *values;
        int *inner_indices;
        int *outer_pointers;

        {
            auto phase_begin = clock::now();

            uint64_t val_bytes = nnz * sizeof(double);
            uint64_t inner_bytes = nnz * sizeof(int);
            uint64_t outer_bytes = (cols + 1) * sizeof(int);
            uint64_t total_bytes = myid == 0 ? val_bytes + inner_bytes + outer_bytes : 0;

            void *A_ptr;
            MPI_Win_allocate_shared(total_bytes, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &A_ptr, &A_win);

            if (myid != 0)
            {
                int disp_unit;
                MPI_Aint sz;
                MPI_Win_shared_query(A_win, 0, &sz, &disp_unit, &A_ptr);
            }

            values = (double *)A_ptr;
            inner_indices = (int *)((char *)A_ptr + val_bytes);
            outer_pointers = (int *)((char *)A_ptr + inner_bytes + val_bytes);

            MPI_Win_fence(0, A_win);

            if (myid == 0)
            {
                std::memcpy(values, Ain.valuePtr(), val_bytes);
                std::memcpy(inner_indices, Ain.innerIndexPtr(), inner_bytes);
                std::memcpy(outer_pointers, Ain.outerIndexPtr(), outer_bytes);
            }

            MPI_Win_fence(0, A_win);

            CPUHYBRID_LOG_INFO("[{}] [create_shared_matrix_window] [{:.6f}]", name(), elapsed_seconds(phase_begin));
        }

        SharedSparseMatrix shared_A(rows, cols, nnz, outer_pointers, inner_indices, values);

        auto phase_begin = clock::now();

        bad_indices_sets.clear();
        bad_indices_arrays.clear();
        ras_target_dofs.clear();
        select_bad_dofs(shared_A);

        if (myid == 0 && contact_patch_schwarz)
        {
            for (const auto &patch : contact_patches)
            {
                bad_indices_sets.emplace_back(patch.begin(), patch.end());
                bad_indices_arrays.emplace_back(patch.begin(), patch.end());
            }

            if (contact_patch_remove_covered)
            {
                remove_covered_contact_patches();
            }
        }

        if (myid == 0 && ras_correction)
        {
            // Snapshot all_bad_dofs into per-dof RAS subdomains now, before
            // the filter/expand/decompose block below can mutate or replace
            // all_bad_dofs -- the same reason contact_patch_schwarz builds
            // its subdomains up front instead of going through that block.
            build_ras_subdomains(shared_A);
        }

        if (myid == 0)
        {
            if (decompose_subdomains && !ras_correction)
            {
                filter_subdomains(shared_A);
            }

            if (expand_subdomains && !ras_correction)
            {
                expand_subdomains_to_strongly_connected(shared_A);
            }

            if (decompose_subdomains && !ras_correction)
            {
                decompose_subdomains_to_disjoint_subsets(shared_A);
            }
            else if (!contact_patch_schwarz && !ras_correction)
            {
                bad_indices_sets.emplace_back(all_bad_dofs.begin(), all_bad_dofs.end());
            }

            load_balance_subdomains();
        }

        share_bad_subdomains();

        if (contact_patch_schwarz)
        {
            compute_overlap_scale();
        }

        factorize_submatrix(shared_A);

        CPUHYBRID_LOG_INFO("[{}] [setup_problematic_dof_precond] [{:.6f}]", name(), elapsed_seconds(phase_begin));

        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
            A = nullptr;
        }

        {
            auto phase_begin = clock::now();
            copy_matrix_to_hypre(shared_A);
            has_matrix_ = true;
            CPUHYBRID_LOG_INFO("[{}] [copy_matrix_to_hypre] [{:.6f}]", name(), elapsed_seconds(phase_begin));
        }

        MPI_Win_free(&A_win);
    }

    ////////////////////////////////////////////////////////////////////////////////

    namespace
    {

        void eigen_to_hypre_par_vec(HYPRE_ParVector &par_x, HYPRE_IJVector &ij_x, const Eigen::VectorXd &x, int start_i, int end_i)
        {
            HYPRE_IJVectorSetValues(ij_x, end_i - start_i + 1, nullptr, x.data() + start_i);
            HYPRE_IJVectorAssemble(ij_x);
            HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);
        }

        void hypre_vec_to_eigen(const HYPRE_IJVector &ij_x, Eigen::Ref<Eigen::VectorXd> x, int start_i, int end_i, int num_procs)
        {
            x.setZero();
            HYPRE_IJVectorGetValues(ij_x, end_i - start_i + 1, nullptr, x.data() + start_i);

            std::vector<int> recv_counts(num_procs);
            std::vector<int> displs(num_procs);

            int local_size = end_i - start_i + 1;
            MPI_Allgather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

            displs[0] = 0;
            for (int i = 1; i < num_procs; ++i)
            {
                displs[i] = displs[i - 1] + recv_counts[i - 1];
            }

            MPI_Allgatherv(x.data() + start_i, local_size, MPI_DOUBLE,
                           x.data(), recv_counts.data(), displs.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
        }

        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 10; // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
            int agg_levels = 1;    // number of aggressive coarsening levels
            double theta = 0.25;   // strength threshold: 0.25, 0.5, 0.8

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i, 0 = classical
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 8;   // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
            int relax_sweeps = 1; // relaxation sweeps on each level

            // Additional options:
            int print_level = 0; // print AMG iterations? 1 = no, 2 = yes
            int max_levels = 25; // max number of levels in AMG hierarchy

            int min_coarse_size = 5;

            HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
            HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);

            // relax_type = 88;
            HYPRE_BoomerAMGSetMinCoarseSize(amg_precond, min_coarse_size);
            // HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_type, 1);
            // HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_type, 2);
            HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_type, 3);
            // HYPRE_BoomerAMGSetDebugFlag(amg_precond, 1);
            // HYPRE_BoomerAMGSetNodal(amg_precond, 0);
            // HYPRE_BoomerAMGSetNodalDiag(amg_precond, 0);
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

    } // anonymous namespace

    ////////////////////////////////////////////////////////////////////////////////

    void CPUHybridSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        if (myid == 0)
        {
            SolverCmd cmd = CMD_SOLVE;
            MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&solver_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

        int problem_size = rhs.size();
        MPI_Bcast(&problem_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Win vec_win;
        void *vec_ptr;
        create_shared_vec(vec_win, vec_ptr, 3 * problem_size);
        SharedVector shared_vec((double *)vec_ptr, 3 * problem_size);

        shared_rhs.resize(my_size());
        shared_result.resize(my_size());
        z1.resize(my_size());
        z2.resize(my_size());
        z3.resize(my_size());
        r.resize(my_size());
        p.resize(my_size());
        buffer.resize(my_size());

        std::vector<int> displs(num_procs);
        std::vector<int> recv_counts(num_procs);

        for (int i = 0; i < num_procs; ++i)
        {
            int local_size = ends[i] - starts[i] + 1;
            recv_counts[i] = local_size;
            displs[i] = starts[i];
        }

        int local_size = my_size();
        MPI_Scatterv(
            rhs.data(),
            recv_counts.data(),
            displs.data(),
            MPI_DOUBLE,
            shared_rhs.data(),
            local_size,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD);

        MPI_Scatterv(
            result.data(),
            recv_counts.data(),
            displs.data(),
            MPI_DOUBLE,
            shared_result.data(),
            local_size,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD);

        HYPRE_ParVector par_b;
        HYPRE_ParVector par_x;
        init_hypre_vectors();

        /* AMG preconditioner */
        HYPRE_Solver precond;

        /* Now set up the AMG preconditioner and specify any parameters */
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
                    assert(block_mapping_.size() == problem_size);

                    // HYPRE_IJMatrixCreate for this rank used the global row range
                    // [starts[myid], ends[myid]], so dof_func must be the matching
                    // *local* slice of the (globally-indexed) mapping -- each rank
                    // only reports the function numbers for the rows it owns.
                    const int n_local = my_size();
                    HYPRE_Int *dof_func = hypre_CTAlloc(HYPRE_Int, n_local, HYPRE_MEMORY_HOST);
                    for (int i = 0; i < n_local; ++i)
                        dof_func[i] = block_mapping_[starts[myid] + i];
                    HYPRE_BoomerAMGSetDofFunc(precond, dof_func);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);

            HYPRE_IJVectorSetData(ij_b, shared_rhs.data());
            HYPRE_IJVectorAssemble(ij_b);
            HYPRE_IJVectorGetObject(ij_b, (void **)&par_b);

            HYPRE_IJVectorSetData(ij_x, shared_result.data());
            HYPRE_IJVectorAssemble(ij_x);
            HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);

            HYPRE_BoomerAMGSetup(precond, parcsr_A, par_b, par_x);
            CPUHYBRID_LOG_INFO("[{}] [amg_setup] [{:.6f}]", name(), elapsed_seconds(phase_begin));
        }

        /* Now setup and solve! */
        {
            auto phase_begin = clock::now();

            pcg_solve(shared_rhs, shared_result, par_b, par_x, precond, shared_vec, vec_win);

            MPI_Gatherv(
                shared_result.data(),
                local_size,
                MPI_DOUBLE,
                result.data(),
                recv_counts.data(),
                displs.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

            Eigen::VectorXd A_times_result;
            matmul(shared_result, buffer);
            buffer = shared_rhs - buffer;
            final_res_norm = sqrt(dot(buffer, buffer));
            CPUHYBRID_LOG_INFO("[{}] [pcg_solve] [{:.6f}] [pcg_iters={}] [residual={}]", name(), elapsed_seconds(phase_begin), num_iterations, final_res_norm);
        }

        /* Destroy preconditioner */
        {
            HYPRE_BoomerAMGDestroy(precond);
            HYPRE_IJVectorDestroy(ij_x);
            HYPRE_IJVectorDestroy(ij_b);
            MPI_Win_free(&vec_win);
        }
    }

    void CPUHybridSolver::pcg_solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond, SharedVector &vec, MPI_Win &vec_win)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        double pre_loop_time;
        double bi_prod, rel_eps, abs_eps, gamma, old_gamma;

        num_iterations = 0;

        r.setZero();
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
                result.setZero();
                MPI_Barrier(MPI_COMM_WORLD);
                return;
            }

            matmul(result, buffer);
            r += rhs - buffer;

            p.setZero();
            z1.setZero();

            {
                custom_mixed_precond_iter(precond, r, z1, vec, vec_win);
            }

            p = z1;

            gamma = dot(r, z1);
            old_gamma = gamma;
            CPUHYBRID_LOG_INFO("[{}] [pre_loop] [{:.6f}] [rhs_norm={}]", name(), elapsed_seconds(phase_begin), sqrt(bi_prod));
        }

        for (int k = 0; k < max_iter_; ++k)
        {
            auto phase_begin = clock::now();
            num_iterations = k + 1;

            matmul(p, buffer);
            double sdotp = dot(p, buffer);

            if (sdotp == 0.0)
            {
                CPUHYBRID_LOG_INFO("[{}] [err_zero_sdotp] [0.000000]", name());
                break;
            }

            double alpha = gamma / sdotp;

            if (alpha <= 0.0)
            {
                CPUHYBRID_LOG_INFO("[{}] [err_negative_alpha] [0.000000]", name());
                break;
            }
            else if (alpha < __DBL_MIN__)
            {
                CPUHYBRID_LOG_INFO("[{}] [err_subnormal_alpha] [0.000000]", name());
                break;
            }

            result += alpha * p;
            r -= alpha * buffer;

            double i_prod = dot(r, r);
            if (rel_eps > 0 && (i_prod / bi_prod) < rel_eps)
            {
                CPUHYBRID_LOG_INFO("[{}] [converged_rel] [0.000000]", name());
                break;
            }

            if (abs_eps > 0 && i_prod < abs_eps)
            {
                CPUHYBRID_LOG_INFO("[{}] [converged_abs] [0.000000]", name());
                break;
            }

            z1.setZero();

            custom_mixed_precond_iter(precond, r, z1, vec, vec_win);

            gamma = dot(r, z1);
            double beta = gamma / old_gamma;
            old_gamma = gamma;

            p = z1 + beta * p;
            CPUHYBRID_LOG_ITER("[{}] [pcg_iter] [{:.6f}] [iter={}] [residual={}]", name(), elapsed_seconds(phase_begin), k, sqrt(i_prod));
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void CPUHybridSolver::custom_mixed_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd &r, Eigen::VectorXd &z, SharedVector &vec, MPI_Win &vec_win)
    {
        z1.setZero();
        z2.setZero();
        z3.setZero();

        if (bad_indices_arrays.size() == 0 || bad_indices_arrays[0].size() == 0)
        {
            amg_precond_iter(precond, r, z1);
            z = z1;
            return;
        }

        if (additive_mode)
        {
            amg_precond_iter(precond, r, z1);
            dss_precond_iter(z3, r, z2, vec, vec_win);
            z = z1 + z2;
        }
        else
        {
            amg_precond_iter(precond, r, z1);
            dss_precond_iter(z1, r, z2, vec, vec_win);
            matmul(z2, buffer);
            Eigen::VectorXd curr_r = r - buffer;
            amg_precond_iter(precond, curr_r, z3);
            z = z2 + z3;
        }
    }

    void CPUHybridSolver::amg_precond_iter(const HYPRE_Solver &precond, Eigen::VectorXd &eigen_b, Eigen::VectorXd &eigen_x)
    {
        auto phase_begin = clock::now();
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_b;

        MPI_Barrier(MPI_COMM_WORLD);

        HYPRE_IJVectorSetData(ij_x, eigen_x.data());
        HYPRE_IJVectorSetData(ij_b, eigen_b.data());
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);
        HYPRE_IJVectorAssemble(ij_b);
        HYPRE_IJVectorGetObject(ij_b, (void **)&par_b);

        HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);
        MPI_Barrier(MPI_COMM_WORLD);
        CPUHYBRID_LOG_INFO("[{}] [amg_v_cycle] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    void CPUHybridSolver::dss_precond_iter(Eigen::VectorXd &z, Eigen::VectorXd &r, Eigen::VectorXd &next_z, SharedVector &vec, MPI_Win &vec_win)
    {
        auto phase_begin = clock::now();
        const int problem_size = vec.size() / 3;

        MPI_Win_fence(0, vec_win);
        for (int i = 0; i < my_size(); ++i)
        {
            vec(starts[myid] + i) = z(i);
            vec(problem_size + starts[myid] + i) = r(i);
            vec(2 * problem_size + starts[myid] + i) = 0.0;
        }
        MPI_Win_fence(0, vec_win);

        // Contact patches may overlap, including patches assigned to
        // different ranks. Rather than have every rank atomically update a
        // shared correction vector, each rank accumulates the subdomains it
        // owns into a private buffer (plain += is safe -- nothing else
        // touches it) and the per-dof sums are then combined once, across
        // all ranks, with a single sum reduction.
        Eigen::VectorXd local_correction;
        if (contact_patch_schwarz)
        {
            local_correction = Eigen::VectorXd::Zero(problem_size);
        }

        int index_counter = 0;
        for (int index : bad_subdomain_assignments[myid])
        {
            auto &subdomain = bad_indices_arrays[index];
            Eigen::VectorXd sub_rhs(subdomain.size());
            Eigen::VectorXd sub_result(subdomain.size());

            for (int i = 0; i < subdomain.size(); ++i)
            {
                sub_rhs(index_mappings[index_counter][subdomain[i]]) = vec(problem_size + subdomain[i]) - global_to_row[subdomain[i]].dot(vec.segment(0, problem_size));
            }

            {
                sub_result = D_solvers[index_counter]->solve(sub_rhs);
            }

            if (ras_correction)
            {
                // Restricted Additive Schwarz: this subdomain exists only to
                // correct the one dof it was built around, even though the
                // solve above covered its whole row neighborhood -- so
                // overlapping subdomains never double-correct a shared dof.
                const int target = ras_target_dofs[index];
                vec(2 * problem_size + target) = sub_result(index_mappings[index_counter][target]);
            }
            else
            {
                for (int i = 0; i < subdomain.size(); ++i)
                {
                    if (contact_patch_schwarz)
                    {
                        local_correction(subdomain[i]) += sub_result(index_mappings[index_counter][subdomain[i]]);
                    }
                    else
                    {
                        vec(2 * problem_size + subdomain[i]) = sub_result(index_mappings[index_counter][subdomain[i]]);
                    }
                }
            }
            ++index_counter;
        }

        if (contact_patch_schwarz)
        {
            // Sum each rank's private contribution into the true overall
            // correction, then damp it by the (global, not per-dof) average
            // amount of patch overlap so overlapping patches don't
            // over-correct on average.
            MPI_Allreduce(MPI_IN_PLACE, local_correction.data(), problem_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            local_correction *= overlap_scale_;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Win_fence(0, vec_win);
        for (int i = 0; i < my_size(); ++i)
        {
            next_z(i) = vec(starts[myid] + i) + (contact_patch_schwarz ? local_correction(starts[myid] + i) : vec(2 * problem_size + starts[myid] + i));
        }
        MPI_Win_fence(0, vec_win);

        CPUHYBRID_LOG_ITER("[{}] [subdomain_solve] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    void CPUHybridSolver::select_bad_dofs(SharedSparseMatrix &sparse_A)
    {
        auto phase_begin = clock::now();

        if (!select_bad_dofs_from_l1_row_norm)
        {
            // AMGF mode: bad DOFs come only from set_problematic_dofs (and,
            // if enabled, contact_patch_schwarz), not from this heuristic.
            return;
        }

        MPI_Win row_norm_win;
        int local_alloc_size = (myid == 0) ? (sparse_A.rows() * sizeof(double)) : 0;
        void *row_norm_ptr;

        MPI_Win_allocate_shared(local_alloc_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &row_norm_ptr, &row_norm_win);
        if (myid != 0)
        {
            int disp_unit;
            MPI_Aint sz;
            MPI_Win_shared_query(row_norm_win, 0, &sz, &disp_unit, &row_norm_ptr);
        }

        SharedVector row_norms((double *)row_norm_ptr, sparse_A.rows());

        MPI_Win_fence(0, row_norm_win);
        for (int i = starts[myid]; i <= ends[myid]; ++i)
        {
            double row_sum = 0;
            for (SharedSparseMatrix::InnerIterator it(sparse_A, i); it; ++it)
            {
                row_sum += std::abs(it.value());
            }
            row_norms(i) = row_sum;
        }
        MPI_Win_fence(0, row_norm_win);

        double global_mean = row_norms.segment(starts[myid], my_size()).sum() / row_norms.size();
        MPI_Allreduce(MPI_IN_PLACE, &global_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double global_var = (row_norms.segment(starts[myid], my_size()).array() - global_mean).square().sum() / row_norms.size();
        MPI_Allreduce(MPI_IN_PLACE, &global_var, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        all_bad_dofs.clear();

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
            if (myid == 0)
            {
                std::vector<std::pair<double, int>> sorted_norms;
                sorted_norms.reserve(row_norms.size());
                for (int i = 0; i < row_norms.size(); ++i)
                {
                    sorted_norms.push_back({row_norms(i), i});
                }

                std::sort(sorted_norms.begin(), sorted_norms.end());
                int N = sorted_norms.size();

                if (N >= 2)
                {
                    if (subdomain_selection_strategy == SubdomainSelectionStrategy::KNEE)
                    {
                        double x1 = 0.0;
                        double y1 = std::log(sorted_norms[0].first + 1e-12);
                        double x2 = N - 1.0;
                        double y2 = std::log(sorted_norms[N - 1].first + 1e-12);

                        for (int i = 1; i < N - 1; ++i)
                        {
                            double x0 = (double)i;
                            double y0 = std::log(sorted_norms[i].first + 1e-12);

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
                            double jump = sorted_norms[i].first - sorted_norms[i - 1].first;
                            if (jump > max_jump)
                            {
                                max_jump = jump;
                                split_idx = i;
                            }
                        }
                    }
                    else if (subdomain_selection_strategy == SubdomainSelectionStrategy::COST)
                    {
                        // TODO: Migrate these to class variables later
                        const double c_pcg_conv = 1.0;
                        const double c_pcg = 1.0;
                        const double c_fac = 1.0;
                        const double c_solve = 1.0;

                        min_cost = std::numeric_limits<double>::max();

                        double lambda_min = sorted_norms[0].first;

                        for (int i = 0; i < N; ++i)
                        {
                            double p = static_cast<double>(N - i);
                            double lambda_max = sorted_norms[i].first;

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

                    for (int i = split_idx; i < N; ++i)
                    {
                        all_bad_dofs.insert(sorted_norms[i].second);
                    }
                }
            }
            break;
        }
        case SubdomainSelectionStrategy::GMM:
        {
            MPI_Win gamma_win;
            void *gamma_ptr;
            MPI_Win_allocate_shared(2 * local_alloc_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &gamma_ptr, &gamma_win);
            if (myid != 0)
            {
                int disp_unit;
                MPI_Aint sz;
                MPI_Win_shared_query(gamma_win, 0, &sz, &disp_unit, &gamma_ptr);
            }

            SharedVector gamma((double *)gamma_ptr, 2 * sparse_A.rows());

            mean_0 = row_norms.minCoeff();
            var_0 = global_var;
            mean_1 = row_norms.maxCoeff();
            var_1 = global_var;
            double w0 = 0.5;
            double w1 = 0.5;
            double var_reg = 1e-6;

            for (gmm_iter = 0; gmm_iter < max_gmm_iterations; ++gmm_iter)
            {
                double log_w0 = std::log(w0);
                double log_w1 = std::log(w1);
                double log_norm_const_0 = -0.5 * std::log(2.0 * M_PI * var_0);
                double log_norm_const_1 = -0.5 * std::log(2.0 * M_PI * var_1);

                auto x = row_norms.segment(starts[myid], my_size()).array();

                Eigen::ArrayXd log_g0 = log_w0 + log_norm_const_0 - 0.5 * (x - mean_0).square() / var_0;
                Eigen::ArrayXd log_g1 = log_w1 + log_norm_const_1 - 0.5 * (x - mean_1).square() / var_1;
                Eigen::ArrayXd max_log_g = log_g0.cwiseMax(log_g1);
                Eigen::ArrayXd log_total = max_log_g + ((log_g0 - max_log_g).exp() + (log_g1 - max_log_g).exp()).log();

                MPI_Win_fence(0, gamma_win);
                gamma.segment(starts[myid], my_size()).array() = (log_g0 - log_total).exp();
                gamma.segment(starts[myid] + row_norms.size(), my_size()).array() = (log_g1 - log_total).exp();
                MPI_Win_fence(0, gamma_win);

                w0 = 1.0 / row_norms.size() * gamma.segment(starts[myid], my_size()).sum();
                w1 = 1.0 / row_norms.size() * gamma.segment(starts[myid] + row_norms.size(), my_size()).sum();

                MPI_Allreduce(MPI_IN_PLACE, &w0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &w1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                double old_mean_0 = mean_0;
                double old_mean_1 = mean_1;
                double old_var_0 = var_0;
                double old_var_1 = var_1;

                mean_0 = (row_norms.segment(starts[myid], my_size()).array() * gamma.segment(starts[myid], my_size()).array()).sum() / (w0 * row_norms.size());
                mean_1 = (row_norms.segment(starts[myid], my_size()).array() * gamma.segment(starts[myid] + row_norms.size(), my_size()).array()).sum() / (w1 * row_norms.size());

                MPI_Allreduce(MPI_IN_PLACE, &mean_0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &mean_1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                var_0 = (gamma.segment(starts[myid], my_size()).array() * (row_norms.segment(starts[myid], my_size()).array() - mean_0).square()).sum() / (w0 * row_norms.size());
                var_1 = (gamma.segment(starts[myid] + row_norms.size(), my_size()).array() * (row_norms.segment(starts[myid], my_size()).array() - mean_1).square()).sum() / (w1 * row_norms.size());

                MPI_Allreduce(MPI_IN_PLACE, &var_0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &var_1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                var_0 += var_reg;
                var_1 += var_reg;

                if (std::abs(mean_0 - old_mean_0) / std::abs(old_mean_0) < gmm_tol && std::abs(mean_1 - old_mean_1) / std::abs(old_mean_1) < gmm_tol && std::abs(var_0 - old_var_0) / std::abs(old_var_0) < gmm_tol && std::abs(var_1 - old_var_1) / std::abs(old_var_1) < gmm_tol)
                {
                    break;
                }
            }

            if (myid == 0)
            {
                if (std::abs(mean_1) / std::abs(mean_0) > gmm_jump_threshold)
                {
                    for (int i = 0; i < row_norms.size(); ++i)
                    {
                        if (gamma(i) < gamma(i + row_norms.size()))
                        {
                            all_bad_dofs.insert(i);
                        }
                    }
                }
            }

            MPI_Win_free(&gamma_win);
            break;
        }
        case SubdomainSelectionStrategy::APOSTERIORI:
            break;
        }

        MPI_Win_free(&row_norm_win);

        switch (subdomain_selection_strategy)
        {
        case SubdomainSelectionStrategy::KNEE:
            CPUHYBRID_LOG_INFO("[{}] [bad_dof_selection] [{}] [strategy=KNEE] [global_mean={}] [global_var={}] [split_idx={}] [max_dist={}] [num_bad_dofs={}]",
                         name(), elapsed_seconds(phase_begin), global_mean, global_var, split_idx, max_dist, all_bad_dofs.size());
            break;
        case SubdomainSelectionStrategy::FD:
            CPUHYBRID_LOG_INFO("[{}] [bad_dof_selection] [{}] [strategy=FD] [global_mean={}] [global_var={}] [split_idx={}] [max_jump={}] [num_bad_dofs={}]",
                         name(), elapsed_seconds(phase_begin), global_mean, global_var, split_idx, max_jump, all_bad_dofs.size());
            break;
        case SubdomainSelectionStrategy::COST:
            CPUHYBRID_LOG_INFO("[{}] [bad_dof_selection] [{}] [strategy=COST] [global_mean={}] [global_var={}] [split_idx={}] [min_cost={}] [num_bad_dofs={}]",
                         name(), elapsed_seconds(phase_begin), global_mean, global_var, split_idx, min_cost, all_bad_dofs.size());
            break;
        case SubdomainSelectionStrategy::GMM:
            CPUHYBRID_LOG_INFO("[{}] [bad_dof_selection] [{}] [strategy=GMM] [global_mean={}] [global_var={}] [mean_0={}] [mean_1={}] [var_0={}] [var_1={}] [gmm_iters={}] [num_bad_dofs={}]",
                         name(), elapsed_seconds(phase_begin), global_mean, global_var, mean_0, mean_1, var_0, var_1, gmm_iter, all_bad_dofs.size());
            break;
        case SubdomainSelectionStrategy::APOSTERIORI:
            break;
        }
    }

    void CPUHybridSolver::factorize_submatrix(SharedSparseMatrix &sparse_A)
    {
        auto phase_begin = clock::now();
        D_solvers.clear();

        for (int i : bad_subdomain_assignments[myid])
        {
            if (bad_indices_sets[i].size() > 1000)
            {
#ifdef POLYSOLVE_WITH_STRUMPACK
                if (use_strumpack)
                {
                    D_solvers.push_back(std::make_unique<StrumpackWrapper>(strumpack_compression, strumpack_compression_rel_tol));
                    continue;
                }
#endif
#if POLYSOLVE_WITH_MKL
                D_solvers.push_back(std::make_unique<EigenWrapper<Eigen::PardisoLDLT<Eigen::SparseMatrix<double>>>>());
#elif POLYSOLVE_WITH_ACCELERATE
                // Uses Apple's Accelerate framework wrapper in Eigen
                D_solvers.push_back(std::make_unique<EigenWrapper<Eigen::AccelerateLDLT<Eigen::SparseMatrix<double>>>>());
#else
                // Fallback if neither high-performance solver is compiled
                D_solvers.push_back(std::make_unique<EigenWrapper<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>>());
#endif
            }
            else
            {
                // Stick to the lightweight solver for smaller subdomains
                D_solvers.push_back(std::make_unique<EigenWrapper<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>>());
            }
        }

        build_index_mappings();

        int i_counter = 0;
        for (int i : bad_subdomain_assignments[myid])
        {
            Eigen::SparseMatrix<double> D;
            assemble_D(i_counter, i, D, sparse_A);
            D_solvers[i_counter]->compute(D);
            ++i_counter;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        CPUHYBRID_LOG_INFO("[{}] [factorize_submatrix] [{}]", name(), elapsed_seconds(phase_begin));
    }

    void CPUHybridSolver::matmul(Eigen::VectorXd &x, Eigen::VectorXd &result)
    {
        auto phase_begin = clock::now();
        result.resize(x.size());
        result.setZero();
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_result;
        HYPRE_IJVectorSetData(ij_x, x.data());
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);
        HYPRE_IJVectorSetData(ij_b, result.data());
        HYPRE_IJVectorAssemble(ij_b);
        HYPRE_IJVectorGetObject(ij_b, (void **)&par_result);
        HYPRE_ParCSRMatrixMatvec(1.0, parcsr_A, par_x, 0.0, par_result);
        CPUHYBRID_LOG_ITER("[{}] [matmul] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    double CPUHybridSolver::dot(Eigen::VectorXd &a, Eigen::VectorXd &b)
    {
        HYPRE_ParVector par_a;
        HYPRE_ParVector par_b;
        HYPRE_IJVectorSetData(ij_x, a.data());
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **)&par_a);
        HYPRE_IJVectorSetData(ij_b, b.data());
        HYPRE_IJVectorAssemble(ij_b);
        HYPRE_IJVectorGetObject(ij_b, (void **)&par_b);
        double result;
        HYPRE_ParVectorInnerProd(par_a, par_b, &result);
        return result;
    }

    void CPUHybridSolver::partition_ranks(const int rows)
    {
        starts.clear();
        ends.clear();
        int local_size = rows / num_procs;
        for (int i = 0; i < num_procs; ++i)
        {
            starts.push_back(i == 0 ? 0 : local_size * i + i);
            ends.push_back(i == (num_procs - 1) ? rows - 1 : starts.back() + local_size);
        }
    }

    void CPUHybridSolver::copy_matrix_to_hypre(SharedSparseMatrix &sparse_A)
    {
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, starts[myid], ends[myid], starts[myid], ends[myid], &A);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        for (HYPRE_Int k = starts[myid]; k <= ends[myid]; ++k)
        {
            HYPRE_Int row[1];
            row[0] = k;
            int counter = 0;
            std::vector<HYPRE_Int> cols;
            std::vector<double> vals;
            for (SharedSparseMatrix::InnerIterator it(sparse_A, k); it; ++it)
            {
                ++counter;
                row[0] = it.col();
                cols.push_back((HYPRE_Int)it.row());
                vals.push_back(it.value());
            }
            HYPRE_Int n_cols[1] = {counter};
            if (counter > 0)
            {
                HYPRE_IJMatrixSetValues(A, 1, n_cols, row, cols.data(), vals.data());
            }
        }
        HYPRE_IJMatrixAssemble(A);
        HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);
    }

    void CPUHybridSolver::copy_matrix_to_hypre(Eigen::SparseMatrix<double> &sparse_A)
    {
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, starts[myid], ends[myid], starts[myid], ends[myid], &A);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        for (HYPRE_Int k = starts[myid]; k <= ends[myid]; ++k)
        {
            HYPRE_Int row[1];
            row[0] = k;
            int counter = 0;
            std::vector<HYPRE_Int> cols;
            std::vector<double> vals;
            for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_A, k); it; ++it)
            {
                ++counter;
                row[0] = it.col();
                cols.push_back((HYPRE_Int)it.row());
                vals.push_back(it.value());
            }
            HYPRE_Int n_cols[1] = {counter};
            HYPRE_IJMatrixSetValues(A, 1, n_cols, row, cols.data(), vals.data());
        }
        HYPRE_IJMatrixAssemble(A);
        HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);
    }

    void CPUHybridSolver::init_hypre_vectors()
    {
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, starts[myid], ends[myid], &ij_x);
        HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
        HYPRE_IJVectorInitializeShell(ij_x);

        hypre_ParVector *par_x = (hypre_ParVector *)hypre_IJVectorObject((hypre_IJVector *)ij_x);
        if (par_x && hypre_ParVectorLocalVector(par_x))
        {
            hypre_Vector *vec_x = hypre_ParVectorLocalVector(par_x);

            hypre_VectorNumVectors(vec_x) = 1;
            hypre_VectorMultiVecStorageMethod(vec_x) = 0;

            hypre_VectorVectorStride(vec_x) = hypre_VectorSize(vec_x);
            hypre_VectorIndexStride(vec_x) = 1;
        }

        HYPRE_IJVectorCreate(MPI_COMM_WORLD, starts[myid], ends[myid], &ij_b);
        HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
        HYPRE_IJVectorInitializeShell(ij_b);

        hypre_ParVector *par_b = (hypre_ParVector *)hypre_IJVectorObject((hypre_IJVector *)ij_b);
        if (par_b && hypre_ParVectorLocalVector(par_b))
        {
            hypre_Vector *vec_b = hypre_ParVectorLocalVector(par_b);

            hypre_VectorNumVectors(vec_b) = 1;
            hypre_VectorMultiVecStorageMethod(vec_b) = 0;

            hypre_VectorVectorStride(vec_b) = hypre_VectorSize(vec_b);
            hypre_VectorIndexStride(vec_b) = 1;
        }
    }

    void CPUHybridSolver::create_shared_vec(MPI_Win &win, void *&base_ptr, int size)
    {
        int local_alloc_size = myid == 0 ? size * sizeof(double) : 0;
        MPI_Win_allocate_shared(local_alloc_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &base_ptr, &win);
        if (myid != 0)
        {
            int disp_unit;
            MPI_Aint sz;
            MPI_Win_shared_query(win, 0, &sz, &disp_unit, &base_ptr);
        }
    }

    void CPUHybridSolver::assemble_D(int bad_i, int i, Eigen::SparseMatrix<double> &D, SharedSparseMatrix &sparse_A)
    {
        D.resize(bad_indices_sets[i].size(), bad_indices_sets[i].size());
        std::vector<Eigen::Triplet<double>> triplets;
        for (int k : bad_indices_sets[i])
        {
            global_to_row[k] = sparse_A.col(k);
            for (SharedSparseMatrix::InnerIterator it(sparse_A, k); it; ++it)
            {
                auto ind_it = index_mappings[bad_i].find(it.row());
                if (ind_it != index_mappings[bad_i].end())
                {
                    triplets.push_back(Eigen::Triplet<double>(index_mappings[bad_i][it.row()], index_mappings[bad_i][it.col()], it.value()));
                }
            }
        }

        D.setFromTriplets(triplets.begin(), triplets.end());
    }

    void CPUHybridSolver::build_index_mappings()
    {
        index_mappings.clear();
        index_mappings.resize(bad_subdomain_assignments[myid].size());

        int i_counter = 0;
        for (int i : bad_subdomain_assignments[myid])
        {
            int j_counter = 0;
            for (auto j : bad_indices_sets[i])
            {
                index_mappings[i_counter][j] = j_counter;
                ++j_counter;
            }
            ++i_counter;
        }
    }

    void CPUHybridSolver::filter_subdomains(SharedSparseMatrix &sparse_A)
    {
        auto phase_begin = clock::now();

        int num_too_small = 0;
        int num_too_large = 0;
        int num_not_poorly_conditioned = 0;
        int original_num_bad_dofs = all_bad_dofs.size();

        int counter = 0;
        std::vector<int> global_to_local(sparse_A.rows(), -1);
        for (auto index : all_bad_dofs)
        {
            global_to_local[index] = counter;
            ++counter;
        }

        hybrid::DisjointSet decomposed_subdomains(all_bad_dofs.size());

        for (int k : all_bad_dofs)
        {
            for (SharedSparseMatrix::InnerIterator it(sparse_A, k); it; ++it)
            {
                if (global_to_local[it.row()] != -1)
                {
                    decomposed_subdomains.union_set(global_to_local[it.row()], global_to_local[it.col()]);
                }
            }
        }

        std::unordered_map<int, std::vector<int>> chosen_sets;
        for (auto index : all_bad_dofs)
        {
            chosen_sets[decomposed_subdomains.find_set(global_to_local[index])].push_back(index);
        }

        all_bad_dofs.clear();

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
                for (SharedSparseMatrix::InnerIterator it(sparse_A, k); it; ++it)
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
                all_bad_dofs.insert(kv.second.begin(), kv.second.end());
                continue;
            }
            ++num_not_poorly_conditioned;
        }

        CPUHYBRID_LOG_INFO("[{}] [subdomain_filtering] [{}] [total_dofs_before={}] [total_dofs_after={}] [num_too_small={}] [num_too_large={}] [num_not_poorly_conditioned={}]",
                     name(), elapsed_seconds(phase_begin), original_num_bad_dofs, all_bad_dofs.size(), num_too_small, num_too_large, num_not_poorly_conditioned);
    }

    void CPUHybridSolver::expand_subdomains_to_strongly_connected(SharedSparseMatrix &sparse_A)
    {
        auto phase_begin = clock::now();
        int num_bad_dofs_before = all_bad_dofs.size();

        std::set<int> new_bad_dofs;

        for (int k : all_bad_dofs)
        {
            for (SharedSparseMatrix::InnerIterator it(sparse_A, k); it; ++it)
            {
                new_bad_dofs.insert(it.row());
            }
        }
        all_bad_dofs = std::move(new_bad_dofs);

        CPUHYBRID_LOG_INFO("[{}] [subdomain_expansion] [{}] [num_dofs_before={}] [num_dofs_after={}]",
                     name(), elapsed_seconds(phase_begin), num_bad_dofs_before, all_bad_dofs.size());
    }

    // A patch fully contained in another contributes nothing that the larger
    // patch doesn't already correct, so it is dropped rather than solved and
    // added redundantly into the overlap sum in dss_precond_iter. Patches are
    // processed smallest-first so each is only ever tested against patches
    // that could actually contain it; ties (identical patches) keep exactly
    // one copy rather than deleting each other.
    void CPUHybridSolver::remove_covered_contact_patches()
    {
        auto phase_begin = clock::now();
        const int num_before = static_cast<int>(bad_indices_sets.size());

        std::vector<int> order(bad_indices_sets.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (bad_indices_sets[a].size() != bad_indices_sets[b].size())
                return bad_indices_sets[a].size() < bad_indices_sets[b].size();
            return a < b;
        });

        std::vector<bool> covered(bad_indices_sets.size(), false);
        for (size_t oi = 0; oi < order.size(); ++oi)
        {
            const int i = order[oi];
            if (covered[i])
                continue;

            for (size_t oj = oi + 1; oj < order.size(); ++oj)
            {
                const int j = order[oj];
                if (covered[j] || bad_indices_sets[i].size() > bad_indices_sets[j].size())
                    continue;

                if (std::includes(bad_indices_sets[j].begin(), bad_indices_sets[j].end(),
                                   bad_indices_sets[i].begin(), bad_indices_sets[i].end()))
                {
                    if (bad_indices_sets[i].size() == bad_indices_sets[j].size())
                    {
                        // Identical patches: keep i (processed first), drop j.
                        covered[j] = true;
                    }
                    else
                    {
                        covered[i] = true;
                        break;
                    }
                }
            }
        }

        std::vector<std::set<int>> kept_sets;
        std::vector<std::vector<int>> kept_arrays;
        kept_sets.reserve(bad_indices_sets.size());
        kept_arrays.reserve(bad_indices_arrays.size());
        for (int i = 0; i < static_cast<int>(bad_indices_sets.size()); ++i)
        {
            if (!covered[i])
            {
                kept_sets.push_back(std::move(bad_indices_sets[i]));
                kept_arrays.push_back(std::move(bad_indices_arrays[i]));
            }
        }
        bad_indices_sets = std::move(kept_sets);
        bad_indices_arrays = std::move(kept_arrays);

        CPUHYBRID_LOG_INFO("[{}] [remove_covered_contact_patches] [{:.6f}] [num_patches_before={}] [num_patches_after={}] [num_removed={}]",
                     name(), elapsed_seconds(phase_begin), num_before, bad_indices_sets.size(), num_before - static_cast<int>(bad_indices_sets.size()));
    }

    // RAS: one subdomain per bad dof, made of every dof with a nonzero entry
    // in that dof's row. sparse_A is column-major, so column k is used as a
    // stand-in for row k -- the same structural-symmetry assumption already
    // relied on elsewhere in this file (e.g. assemble_D's global_to_row).
    // Unlike the other subdomain-construction paths, the subdomain a dof
    // lands in here is never merged, filtered, or expanded: it is exactly the
    // dof's own sparsity neighborhood, and dss_precond_iter restricts its
    // correction to that one owning dof (ras_target_dofs), so overlapping
    // subdomains never double-correct a dof the way additive Schwarz would.
    void CPUHybridSolver::build_ras_subdomains(SharedSparseMatrix &sparse_A)
    {
        auto phase_begin = clock::now();

        for (int d : all_bad_dofs)
        {
            std::set<int> subdomain;
            for (SharedSparseMatrix::InnerIterator it(sparse_A, d); it; ++it)
            {
                subdomain.insert(it.row());
            }
            subdomain.insert(d);

            bad_indices_sets.push_back(subdomain);
            bad_indices_arrays.emplace_back(subdomain.begin(), subdomain.end());
            ras_target_dofs.push_back(d);
        }

        CPUHYBRID_LOG_INFO("[{}] [build_ras_subdomains] [{:.6f}] [num_bad_dofs={}] [num_subdomains={}]",
                     name(), elapsed_seconds(phase_begin), all_bad_dofs.size(), bad_indices_sets.size());
    }

    void CPUHybridSolver::decompose_subdomains_to_disjoint_subsets(SharedSparseMatrix &sparse_A)
    {
        auto phase_begin = clock::now();
        int counter = 0;
        std::vector<int> global_to_local(sparse_A.rows(), -1);
        for (auto index : all_bad_dofs)
        {
            global_to_local[index] = counter;
            ++counter;
        }

        hybrid::DisjointSet decomposed_subdomains(all_bad_dofs.size());

        for (int k : all_bad_dofs)
        {
            for (SharedSparseMatrix::InnerIterator it(sparse_A, k); it; ++it)
            {
                if (global_to_local[it.row()] != -1)
                {
                    decomposed_subdomains.union_set(global_to_local[it.row()], global_to_local[it.col()]);
                }
            }
        }

        std::unordered_map<int, std::vector<int>> chosen_sets;
        for (auto index : all_bad_dofs)
        {
            chosen_sets[decomposed_subdomains.find_set(global_to_local[index])].push_back(index);
        }

        bad_indices_sets.clear();

        for (auto &kv : chosen_sets)
        {
            if (kv.second.size() > max_subdomain_size)
            {
                continue;
            }
            bad_indices_sets.emplace_back(kv.second.begin(), kv.second.end());
        }
        CPUHYBRID_LOG_INFO("[{}] [subdomain_decomposition] [{}] [num_subdomains={}] ",
                     name(), elapsed_seconds(phase_begin), bad_indices_sets.size());
    }

    void CPUHybridSolver::share_bad_subdomains()
    {
        auto phase_begin = clock::now();
        int num_subdomains;
        if (myid == 0)
        {
            num_subdomains = bad_indices_sets.size();
            MPI_Bcast(&num_subdomains, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (bad_indices_sets.size() > 0)
            {
                for (int i = 0; i < num_subdomains; ++i)
                {
                    int num_indices = bad_indices_sets[i].size();
                    MPI_Bcast(&num_indices, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    std::vector<int> subdomain_vec;
                    for (auto index : bad_indices_sets[i])
                    {
                        subdomain_vec.push_back(index);
                    }
                    MPI_Bcast(subdomain_vec.data(), num_indices, MPI_INT, 0, MPI_COMM_WORLD);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
            }
            for (int i = 0; i < num_procs; ++i)
            {
                int local_size = bad_subdomain_assignments[i].size();
                MPI_Bcast(&local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(bad_subdomain_assignments[i].data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
            MPI_Bcast(&num_subdomains, 1, MPI_INT, 0, MPI_COMM_WORLD);
            bad_indices_sets.clear();
            bad_indices_sets.resize(num_subdomains);
            for (int i = 0; i < num_subdomains; ++i)
            {
                int num_indices;
                MPI_Bcast(&num_indices, 1, MPI_INT, 0, MPI_COMM_WORLD);
                std::vector<int> subdomain_vec;
                subdomain_vec.resize(num_indices);
                MPI_Bcast(subdomain_vec.data(), num_indices, MPI_INT, 0, MPI_COMM_WORLD);
                for (auto index : subdomain_vec)
                {
                    bad_indices_sets[i].insert(index);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
            bad_subdomain_assignments.resize(num_procs);
            for (int i = 0; i < num_procs; ++i)
            {
                int local_size;
                MPI_Bcast(&local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
                bad_subdomain_assignments[i].resize(local_size);
                MPI_Bcast(bad_subdomain_assignments[i].data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
            }
        }

        if (ras_correction)
        {
            int ras_size = (myid == 0) ? static_cast<int>(ras_target_dofs.size()) : 0;
            MPI_Bcast(&ras_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (myid != 0)
            {
                ras_target_dofs.resize(ras_size);
            }
            if (ras_size > 0)
            {
                MPI_Bcast(ras_target_dofs.data(), ras_size, MPI_INT, 0, MPI_COMM_WORLD);
            }
        }

        bad_indices_arrays.clear();
        bad_indices_arrays.resize(bad_indices_sets.size());
        for (int i = 0; i < bad_indices_sets.size(); ++i)
        {
            bad_indices_arrays[i].reserve(bad_indices_sets[i].size());
            for (auto index : bad_indices_sets[i])
            {
                bad_indices_arrays[i].push_back(index);
            }
        }
        CPUHYBRID_LOG_INFO("[{}] [share_bad_subdomains] [{}] ",
                     name(), elapsed_seconds(phase_begin));
    }

    // Under contact_patch_schwarz, bad_indices_sets (the contact patches) may
    // overlap, so a covered dof can belong to more than one patch. Compute a
    // single global scale factor -- 1 / (average number of patches per
    // covered dof) -- used to damp the summed correction in dss_precond_iter
    // so overlapping patches don't over-correct on average. bad_indices_sets
    // is identical on every rank by this point (share_bad_subdomains already
    // broadcast it), so no communication is needed here.
    void CPUHybridSolver::compute_overlap_scale()
    {
        std::unordered_map<int, int> dof_patch_count;
        for (const auto &subdomain : bad_indices_sets)
        {
            for (int dof : subdomain)
            {
                ++dof_patch_count[dof];
            }
        }

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

        CPUHYBRID_LOG_INFO("[{}] [contact_patch_overlap] [avg_overlap={}] [overlap_scale={}]", name(), avg_overlap, overlap_scale_);
    }

    void CPUHybridSolver::load_balance_subdomains()
    {
        auto phase_begin = clock::now();
        bad_subdomain_assignments.clear();
        bad_subdomain_assignments.resize(num_procs);

        std::vector<std::pair<int, int>> subdomain_sizes;
        subdomain_sizes.reserve(bad_indices_sets.size());

        for (auto &subdomain : bad_indices_sets)
        {
            subdomain_sizes.push_back(std::make_pair(subdomain_sizes.size(), subdomain.size()));
        }

        std::sort(subdomain_sizes.begin(), subdomain_sizes.end(), [](const std::pair<int, int> &l, const std::pair<int, int> &r) { return l.second > r.second; });
        std::vector<int> assigned_sizes(num_procs, 0);

        int total_bad_dofs = 0;
        for (auto [i, size] : subdomain_sizes)
        {
            int min_size = assigned_sizes[0];
            total_bad_dofs += size;
            int chosen_proc = 0;
            for (int pi = 1; pi < num_procs; ++pi)
            {
                if (assigned_sizes[pi] < min_size)
                {
                    min_size = assigned_sizes[pi];
                    chosen_proc = pi;
                }
            }
            bad_subdomain_assignments[chosen_proc].push_back(i);
            assigned_sizes[chosen_proc] += size;
        }

        const int max_size = subdomain_sizes.size() > 0 ? subdomain_sizes.front().second : 0;
        const int min_size = subdomain_sizes.size() > 0 ? subdomain_sizes.back().second : 0;

        CPUHYBRID_LOG_INFO("[{}] [subdomain_load_balance] [{}] [max_size={}] [min_size={}] [total_dofs={}]",
                     name(), elapsed_seconds(phase_begin), max_size, min_size, total_bad_dofs);
    }

    ////////////////////////////////////////////////////////////////////////////////

    // Worker ranks: service the driver's commands until CMD_EXIT, then return
    // so the thread can be joined. Under mpirun this was the process's whole
    // life; here it is one thread's.
    static int rank_worker_entry(void *)
    {
        CPUHybridSolver::worker_loop_flag() = true;
        if (!HYPRE_Initialized())
        {
            HYPRE_Initialize();
        }
        Eigen::setNbThreads(1);
        HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
        HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
        CPUHybridSolver::run_worker_loop();
        return 0;
    }

    void CPUHybridSolver::ensure_ranks()
    {
        if (is_running_worker_loop)
        {
            return;
        }
        if (rank_team)
        {
            ++live_solvers;
            return;
        }
        // 0 means the default: NANOMPI_NUM_RANKS if set, else cores online
        if (nanompi_team_start(0, rank_worker_entry, nullptr, &rank_team) != 0)
        {
            throw std::runtime_error("polysolve: could not start nano-mpi ranks");
        }
        ++live_solvers;
    }

    // Tell the workers to leave their command loop and join them, which tears
    // the rank universe down and puts COMM_WORLD back to a single rank -- which
    // matters, because a plain HypreSolver constructed afterwards is a lone
    // thread and would otherwise be one rank of a world that expects several.
    void CPUHybridSolver::release_ranks()
    {
        if (is_running_worker_loop || !rank_team)
        {
            return;
        }
        if (--live_solvers > 0)
        {
            return;
        }
        SolverCmd cmd = CMD_EXIT;
        MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
        nanompi_team_join(rank_team);
        rank_team = nullptr;
    }

    void CPUHybridSolver::run_worker_loop()
    {
        bool running = true;
        while (running)
        {
            SolverCmd cmd;
            MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (cmd == CMD_EXIT)
            {
                running = false;
                break;
            }

            int id;
            MPI_Bcast(&id, 1, MPI_INT, 0, MPI_COMM_WORLD);

            switch (cmd)
            {
            case CMD_CREATE:
            {
                worker_registry[id] = std::make_unique<CPUHybridSolver>();
                worker_registry[id]->solver_id = id;
                break;
            }
            case CMD_SET_PARAMETERS:
            {
                json dummy_params;
                worker_registry[id]->set_parameters(dummy_params);
                break;
            }
            case CMD_SET_BLOCK_SIZE:
            {
                int dummy_block_size = 0;
                worker_registry[id]->set_block_size(dummy_block_size);
                break;
            }
            case CMD_SET_BLOCK_MAPPING:
            {
                Eigen::VectorXi dummy_block_mapping;
                worker_registry[id]->set_block_mapping(dummy_block_mapping);
                break;
            }
            case CMD_FACTORIZE:
            {
                StiffnessMatrix dummy_A;
                worker_registry[id]->factorize(dummy_A);
                break;
            }
            case CMD_SOLVE:
            {
                Eigen::VectorXd dummy_b, dummy_x;
                worker_registry[id]->solve(dummy_b, dummy_x);
                break;
            }
            case CMD_DESTROY:
            {
                worker_registry.erase(id);
                break;
            }
            }
        }
    }

    CPUHybridSolver::~CPUHybridSolver()
    {
        if (myid == 0)
        {
            SolverCmd cmd = CMD_DESTROY;
            MPI_Bcast(&cmd, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&solver_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
            A = nullptr;
        }

        if (myid == 0)
        {
            release_ranks();
        }
    }

} // namespace polysolve::linear