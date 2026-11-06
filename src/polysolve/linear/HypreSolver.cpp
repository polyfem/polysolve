#ifdef POLYSOLVE_WITH_HYPRE

////////////////////////////////////////////////////////////////////////////////
#include "HypreSolver.hpp"

#include <HYPRE_krylov.h>
#include <HYPRE_utilities.h>
// Internal header, needed for hypre_CTAlloc/hypre_TMemcpy so the dof_func array
// handed to HYPRE_BoomerAMGSetDofFunc uses the allocator HYPRE itself frees it with.
#include "_hypre_utilities.h"

// Split the rows across the nano-mpi ranks when hypre is built with MPI, which
// in polysolve always means nano-mpi.
#if defined(POLYSOLVE_WITH_MPI) && defined(HYPRE_ENABLE_MPI)
#define POLYSOLVE_HYPRE_ON_RANKS
#include "mpi_utils/RankTeam.hpp"
#endif

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <stdexcept>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve::linear
{

    ////////////////////////////////////////////////////////////////////////////////

    namespace
    {
        // Communicator for a solve that runs on the calling thread alone.
        MPI_Comm single_rank_comm()
        {
#ifdef HYPRE_ENABLE_MPI
            // Not MPI_COMM_WORLD: while nano-mpi ranks exist that is their world,
            // and this thread need not be part of it.
            return MPI_COMM_SELF;
#else
            return 0; // ignored by sequential hypre
#endif
        }

        // hypre keeps its memory location and execution policy in (per-thread)
        // globals, and the other hypre-based solvers set them too. So they are set
        // for the duration of each call rather than once in the constructor.
        class HypreExecution
        {
        public:
            explicit HypreExecution(const bool gpu)
            {
                if (!HYPRE_Initialized())
                {
                    HYPRE_Initialize();
                }

                HYPRE_GetMemoryLocation(&memory_);
                HYPRE_GetExecutionPolicy(&policy_);
#ifdef HYPRE_USING_GPU
                if (gpu)
                {
                    HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
                    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
                    // Same choices as GPUHybridSolver.
                    HYPRE_SetSpGemmUseVendor(0);
                    HYPRE_SetUseGpuRand(1);
                    return;
                }
#else
                assert(!gpu);
                (void)gpu;
#endif
                HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
                HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
            }

            ~HypreExecution()
            {
                HYPRE_SetMemoryLocation(memory_);
                HYPRE_SetExecutionPolicy(policy_);
            }

            HypreExecution(const HypreExecution &) = delete;
            HypreExecution &operator=(const HypreExecution &) = delete;

        private:
            HYPRE_MemoryLocation memory_;
            HYPRE_ExecutionPolicy policy_;
        };

        // A vector over this rank's rows [ilower, iupper], filled from values (this
        // rank's slice, on the host) and moved to the GPU if gpu is set.
        HYPRE_IJVector make_vector(MPI_Comm comm, const HYPRE_BigInt ilower, const HYPRE_BigInt iupper, const double *values, const bool gpu)
        {
            HYPRE_IJVector v;
            HYPRE_IJVectorCreate(comm, ilower, iupper, &v);
            HYPRE_IJVectorSetObjectType(v, HYPRE_PARCSR);
            HYPRE_IJVectorInitialize_v2(v, HYPRE_MEMORY_HOST);
            HYPRE_IJVectorSetValues(v, HYPRE_Int(iupper - ilower + 1), nullptr, values);
            HYPRE_IJVectorAssemble(v);
            if (gpu)
            {
                HYPRE_IJVectorMigrate(v, HYPRE_MEMORY_DEVICE);
            }
            return v;
        }

        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 10; // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
            int agg_levels = 1;    // number of aggressive coarsening levels
            double theta = 0.5;    // strength threshold: 0.25, 0.5, 0.8

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i, 0 = classical
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 8;   // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
            int relax_sweeps = 1; // relaxation sweeps on each level

            // Additional options:
            int print_level = 0; // print AMG iterations? 1 = no, 2 = yes
            int max_levels = 25; // max number of levels in AMG hierarchy

            HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
            HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
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

        // Options hypre can run on the GPU; the same as GPUHybridSolver's.
        void HypreBoomerAMG_SetGPUOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 8; // 8 = PMIS
            int agg_levels = 1;   // number of aggressive coarsening levels
            double theta = 0.25;  // strength threshold

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 18;  // 18 = l1-Jacobi
            int relax_sweeps = 1; // relaxation sweeps on each level

            // Additional options:
            int print_level = 0;
            int max_levels = 25;
            int min_coarse_size = 5;

            HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
            HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
            HYPRE_BoomerAMGSetRelaxOrder(amg_precond, 0);
            HYPRE_BoomerAMGSetRAP2(amg_precond, 1);
            HYPRE_BoomerAMGSetKeepTranspose(amg_precond, 1);
            HYPRE_BoomerAMGSetMinCoarseSize(amg_precond, min_coarse_size);
            // the coarsest level too, rather than the default Gaussian elimination
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

        void HypreBoomerAMG_SetElasticityOptions(HYPRE_Solver &amg_precond, int dim, double theta, bool nodal_coarsening)
        {
            // Make sure the systems AMG options are set
            HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

            // More robust options with respect to convergence
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);

            if (nodal_coarsening)
            {
                // See hypre's new_ij driver and the paper for descriptions.
                int nodal = 4;        // strength reduction norm: 1, 3 or 4
                int nodal_diag = 1;   // diagonal in strength matrix: 0, 1 or 2
                int relax_coarse = 8; // smoother on the coarsest grid: 8, 99 or 29

                HYPRE_BoomerAMGSetNodal(amg_precond, nodal);
                HYPRE_BoomerAMGSetNodalDiag(amg_precond, nodal_diag);
                HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_coarse, 3);
            }
        }

#ifdef POLYSOLVE_HYPRE_ON_RANKS
        const char *const wrong_thread_message =
            "Hypre: this solver's matrix is split across nano-mpi ranks that another thread drives; "
            "use the solver from the thread that first factorized it";
#endif

    } // anonymous namespace

    ////////////////////////////////////////////////////////////////////////////////

    HypreSolver::HypreSolver()
        : comm_(single_rank_comm())
    {
        if (!HYPRE_Initialized())
        {
            HYPRE_Initialize();
        }
    }

    // Set solver parameters
    void HypreSolver::set_parameters(const json &params)
    {
        if (params.contains("Hypre"))
        {
            if (params["Hypre"].contains("max_iter"))
            {
                max_iter_ = params["Hypre"]["max_iter"];
            }
            if (params["Hypre"].contains("pre_max_iter"))
            {
                pre_max_iter_ = params["Hypre"]["pre_max_iter"];
            }
            if (params["Hypre"].contains("tolerance"))
            {
                conv_tol_ = params["Hypre"]["tolerance"];
            }
            if (params["Hypre"].contains("theta"))
            {
                theta = params["Hypre"]["theta"];
            }
            if (params["Hypre"].contains("nodal_coarsening"))
            {
                nodal_coarsening = params["Hypre"]["nodal_coarsening"];
            }
            if (params["Hypre"].contains("interp_rbms"))
            {
                interp_rbms = params["Hypre"]["interp_rbms"];
            }
            if (params["Hypre"].contains("use_gpu"))
            {
                use_gpu_ = params["Hypre"]["use_gpu"];
            }
        }

#ifndef HYPRE_USING_GPU
        if (use_gpu_)
        {
            use_gpu_ = false;
            throw std::runtime_error("Hypre: use_gpu needs polysolve built with POLYSOLVE_WITH_CUDA=ON");
        }
#endif
        if (interp_rbms)
        {
            spdlog::warn("Hypre: ignoring interp_rbms, which needs the node positions and the solver is not given them");
        }
        if (use_gpu_ && nodal_coarsening)
        {
            spdlog::warn("Hypre: ignoring nodal_coarsening on the GPU, where hypre does not implement it");
        }
    }

    void HypreSolver::set_block_size(int block_size)
    {
        dimension_ = block_size;
    }

    void HypreSolver::set_block_mapping(const Eigen::VectorXi &block_mapping)
    {
        block_mapping_ = block_mapping;
    }

    void HypreSolver::get_info(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
        params["num_ranks"] = rank_matrices_.empty() ? 1 : int(rank_matrices_.size());
        params["on_gpu"] = on_gpu_;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void HypreSolver::factorize(const StiffnessMatrix &Ain)
    {
        assert(precond_num_ > 0);
        assert(Ain.rows() == Ain.cols());

        destroy_matrix();
        place_matrix();

#ifdef POLYSOLVE_HYPRE_ON_RANKS
        const int num_ranks = uses_ranks_ ? mpi_utils::RankTeam::size() : 1;
#else
        const int num_ranks = 1;
#endif
        partition_rows(HYPRE_BigInt(Ain.rows()), num_ranks);
        on_each_rank([&](const int rank) { factorize_rank(Ain, rank); });
    }

    // Decides where the next matrix goes: the GPU, the nano-mpi ranks, or the
    // calling thread alone.
    void HypreSolver::place_matrix()
    {
        on_gpu_ = use_gpu_;
#ifdef POLYSOLVE_HYPRE_ON_RANKS
        if (on_gpu_)
        {
            // a GPU solve is a single rank
            release_ranks();
        }
        else if (!uses_ranks_)
        {
            // False on a worker rank, or when another thread drives the ranks;
            // then this thread solves alone.
            uses_ranks_ = mpi_utils::RankTeam::acquire();
        }
        comm_ = uses_ranks_ ? MPI_COMM_WORLD : single_rank_comm();
#else
        comm_ = single_rank_comm();
#endif
    }

    // Contiguous, balanced row ranges, one per rank. They are made of whole
    // blocks, so that no node is split between two ranks (nodal coarsening
    // requires that).
    void HypreSolver::partition_rows(const HYPRE_BigInt rows, const int num_ranks)
    {
        const long long block = (dimension_ > 1 && rows % dimension_ == 0) ? dimension_ : 1;
        const long long num_blocks = rows / block;

        rank_matrices_.assign(num_ranks, RankMatrix());
        for (int r = 0; r < num_ranks; ++r)
        {
            rank_matrices_[r].ilower = HYPRE_BigInt(num_blocks * r / num_ranks * block);
            rank_matrices_[r].iupper = HYPRE_BigInt(num_blocks * (r + 1) / num_ranks * block - 1);
        }
    }

    // Runs fn(rank) for every rank the matrix is split over, each on its own
    // rank, and returns when all are done.
    void HypreSolver::on_each_rank(const std::function<void(int)> &fn)
    {
#ifdef POLYSOLVE_HYPRE_ON_RANKS
        if (uses_ranks_)
        {
            if (!mpi_utils::RankTeam::is_driver())
            {
                throw std::runtime_error(wrong_thread_message);
            }
            // The ranks are threads of this process, so they read the caller's
            // matrix and vectors, and write their slice of the result, in place.
            mpi_utils::RankTeam::run([&fn] {
                int rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                fn(rank);
            });
            return;
        }
#endif
        fn(0);
    }

    void HypreSolver::factorize_rank(const StiffnessMatrix &Ain, const int rank)
    {
        HypreExecution execution(on_gpu_);
        RankMatrix &m = rank_matrices_[rank];
        const HYPRE_Int local_rows = HYPRE_Int(m.iupper - m.ilower + 1);

        HYPRE_IJMatrixCreate(comm_, m.ilower, m.iupper, m.ilower, m.iupper, &m.A);
        HYPRE_IJMatrixSetObjectType(m.A, HYPRE_PARCSR);
        // Assembled on the host; moved to the GPU afterwards in one copy if needed.
        HYPRE_IJMatrixInitialize_v2(m.A, HYPRE_MEMORY_HOST);

        if (local_rows > 0)
        {
            // Ain is symmetric, so its column k is row k: this rank's rows are
            // the columns [ilower, iupper], read straight from Ain in one pass.
            std::vector<HYPRE_Int> n_cols(local_rows);
            std::vector<HYPRE_BigInt> rows(local_rows);
            std::vector<HYPRE_BigInt> cols;
            std::vector<HYPRE_Complex> vals;
            const double expected_nnz = double(Ain.nonZeros()) * local_rows / std::max<Eigen::Index>(Ain.rows(), 1);
            cols.reserve(size_t(expected_nnz));
            vals.reserve(size_t(expected_nnz));

            for (HYPRE_Int i = 0; i < local_rows; ++i)
            {
                const HYPRE_BigInt k = m.ilower + i;
                HYPRE_Int count = 0;
                for (StiffnessMatrix::InnerIterator it(Ain, k); it; ++it, ++count)
                {
                    cols.push_back(HYPRE_BigInt(it.row()));
                    vals.push_back(it.value());
                }
                rows[i] = k;
                n_cols[i] = count;
            }

            HYPRE_IJMatrixSetValues(m.A, local_rows, n_cols.data(), rows.data(), cols.data(), vals.data());
        }

        HYPRE_IJMatrixAssemble(m.A);
        if (on_gpu_)
        {
            HYPRE_IJMatrixMigrate(m.A, HYPRE_MEMORY_DEVICE);
        }
        HYPRE_IJMatrixGetObject(m.A, (void **)&m.parcsr_A);
    }

    ////////////////////////////////////////////////////////////////////////////////

    void HypreSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        assert(!rank_matrices_.empty());
        assert(result.size() == rhs.size());
        assert(rhs.size() == rank_matrices_.back().iupper + 1);

        on_each_rank([&](const int rank) { solve_rank(rhs.data(), result.data(), rank); });
    }

    // One rank's part of the solve: rhs and x are the whole vectors; this rank
    // reads its rows of both and writes its rows of x.
    void HypreSolver::solve_rank(const double *rhs, double *x, const int rank)
    {
        HypreExecution execution(on_gpu_);
        const RankMatrix &m = rank_matrices_[rank];
        const HYPRE_Int local_rows = HYPRE_Int(m.iupper - m.ilower + 1);

        HYPRE_IJVector b = make_vector(comm_, m.ilower, m.iupper, rhs + m.ilower, on_gpu_);
        HYPRE_IJVector ij_x = make_vector(comm_, m.ilower, m.iupper, x + m.ilower, on_gpu_);
        HYPRE_ParVector par_b;
        HYPRE_ParVector par_x;
        HYPRE_IJVectorGetObject(b, (void **)&par_b);
        HYPRE_IJVectorGetObject(ij_x, (void **)&par_x);

        /* PCG with AMG preconditioner */

        /* Create solver */
        HYPRE_Solver solver, precond;
        HYPRE_ParCSRPCGCreate(comm_, &solver);

        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_PCGSetMaxIter(solver, max_iter_); /* max iterations */
        HYPRE_PCGSetTol(solver, conv_tol_);     /* conv. tolerance */
        HYPRE_PCGSetTwoNorm(solver, 1);         /* use the two norm as the stopping criteria */
        // HYPRE_PCGSetPrintLevel(solver, 2); /* print solve info */
        HYPRE_PCGSetLogging(solver, 1); /* needed to get run info later */

        /* Now set up the AMG preconditioner and specify any parameters */
        HYPRE_BoomerAMGCreate(&precond);
        if (on_gpu_)
        {
            HypreBoomerAMG_SetGPUOptions(precond);
        }
        else
        {
            HypreBoomerAMG_SetDefaultOptions(precond);
        }

        if (dimension_ > 1)
        {
            HypreBoomerAMG_SetElasticityOptions(precond, dimension_, theta, nodal_coarsening && !on_gpu_);

            if (block_mapping_.size() > 0)
            {
                // This rank's slice of the global mapping. HYPRE takes ownership
                // of the array and frees it with the AMG preconditioner, so it is
                // allocated with HYPRE's own allocator, in the memory the matrix
                // lives in.
                const HYPRE_MemoryLocation location = on_gpu_ ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;
                std::vector<HYPRE_Int> local_dof_func(local_rows);
                for (HYPRE_Int i = 0; i < local_rows; ++i)
                {
                    local_dof_func[i] = block_mapping_[m.ilower + i];
                }
                HYPRE_Int *dof_func = hypre_CTAlloc(HYPRE_Int, local_rows, location);
                hypre_TMemcpy(dof_func, local_dof_func.data(), HYPRE_Int, local_rows, location, HYPRE_MEMORY_HOST);
                HYPRE_BoomerAMGSetDofFunc(precond, dof_func);
            }
        }

        /* Set the PCG preconditioner */
        HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, precond);

        /* Now setup and solve! */
        HYPRE_ParCSRPCGSetup(solver, m.parcsr_A, par_b, par_x);
        HYPRE_ParCSRPCGSolve(solver, m.parcsr_A, par_b, par_x);

        /* Run info - needed logging turned on. The same on every rank. */
        if (rank == 0)
        {
            HYPRE_PCGGetNumIterations(solver, &num_iterations);
            HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
        }

        /* Destroy solver and preconditioner */
        HYPRE_BoomerAMGDestroy(precond);
        HYPRE_ParCSRPCGDestroy(solver);

        if (on_gpu_)
        {
            HYPRE_IJVectorMigrate(ij_x, HYPRE_MEMORY_HOST);
        }
        HYPRE_IJVectorGetValues(ij_x, local_rows, nullptr, x + m.ilower);

        HYPRE_IJVectorDestroy(b);
        HYPRE_IJVectorDestroy(ij_x);
    }

    ////////////////////////////////////////////////////////////////////////////////

    void HypreSolver::destroy_matrix()
    {
        if (rank_matrices_.empty())
        {
            return;
        }

        on_each_rank([&](const int rank) {
            RankMatrix &m = rank_matrices_[rank];
            if (m.A)
            {
                HypreExecution execution(on_gpu_);
                HYPRE_IJMatrixDestroy(m.A);
                m.A = nullptr;
                m.parcsr_A = nullptr;
            }
        });
        rank_matrices_.clear();
    }

    void HypreSolver::release_ranks()
    {
#ifdef POLYSOLVE_HYPRE_ON_RANKS
        if (uses_ranks_)
        {
            if (!mpi_utils::RankTeam::is_driver())
            {
                throw std::runtime_error(wrong_thread_message);
            }
            mpi_utils::RankTeam::release();
            uses_ranks_ = false;
        }
#endif
    }

    HypreSolver::~HypreSolver()
    {
        try
        {
            destroy_matrix();
            release_ranks();
        }
        catch (const std::exception &e)
        {
            // Only when destroyed from a thread that cannot reach the ranks
            // holding the matrix; all that can be done from here is leave it.
            spdlog::error("{}", e.what());
        }
    }

} // namespace polysolve::linear

#endif
