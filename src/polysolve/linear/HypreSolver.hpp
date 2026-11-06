#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <functional>
#include <vector>

#include <HYPRE_utilities.h>
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

////////////////////////////////////////////////////////////////////////////////
//
// https://computation.llnl.gov/sites/default/files/public/hypre-2.11.2_usr_manual.pdf
// https://github.com/LLNL/hypre/blob/v2.14.0/docs/HYPRE_usr_manual.pdf
//
// PCG preconditioned with BoomerAMG. Where it runs:
//
//  - On the CPU, built with POLYSOLVE_WITH_MPI: the rows are split across the
//    nano-mpi ranks, which are threads of this process (see RankTeam), and hypre
//    solves in parallel. NANOMPI_NUM_RANKS sets the number of ranks; the default
//    is the number of cores.
//  - On the CPU otherwise: on the calling thread.
//  - On the GPU, with the "use_gpu" option (needs POLYSOLVE_WITH_CUDA): on the
//    current CUDA device, as a single rank. One GPU on one node; the nano-mpi
//    ranks are not used.
//

namespace polysolve::linear
{

    class HypreSolver : public Solver
    {

    public:
        HypreSolver();
        ~HypreSolver();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(HypreSolver)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Set block size for multigrid solvers
        virtual void set_block_size(int block_size) override;

        // Set the function (block) assigned to each row for multigrid solvers
        virtual void set_block_mapping(const Eigen::VectorXi &block_mapping) override;

        // Retrieve information about the last solve
        virtual void get_info(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override { precond_num_ = precond_num; }

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "Hypre"; }

        virtual void set_tolerance(const double tol) override { conv_tol_ = tol; }

    protected:
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 1000;
        int pre_max_iter_ = 1;
        double conv_tol_ = 1e-10;

        // solver tuning options
        double theta = 0.5;
        bool nodal_coarsening = false;
        bool interp_rbms = false;

        // Solve on the GPU instead of the CPU.
        bool use_gpu_ = false;

        HYPRE_Int num_iterations = 0;
        HYPRE_Complex final_res_norm = 0;

    private:
        // The rows [ilower, iupper] of the matrix that one rank owns.
        struct RankMatrix
        {
            HYPRE_BigInt ilower = 0;
            HYPRE_BigInt iupper = -1;
            HYPRE_IJMatrix A = nullptr;
            HYPRE_ParCSRMatrix parcsr_A = nullptr;
        };

        int precond_num_ = 0;

        // Optional per-row function (block) assignment for HYPRE_BoomerAMGSetDofFunc,
        // indexed by global row. Empty means the default interleaved mapping
        // (row i belongs to function i % dimension_).
        Eigen::VectorXi block_mapping_;

        // The current matrix: one entry per rank it is split over, and rank r
        // only touches entry r. Empty when there is no matrix.
        std::vector<RankMatrix> rank_matrices_;
        MPI_Comm comm_;
        bool on_gpu_ = false;   // the matrix lives on the GPU
        bool uses_ranks_ = false; // holds a reference on the nano-mpi ranks and splits the matrix over them

        void place_matrix();
        void partition_rows(HYPRE_BigInt rows, int num_ranks);
        void on_each_rank(const std::function<void(int)> &fn);
        void factorize_rank(const StiffnessMatrix &A, int rank);
        void solve_rank(const double *rhs, double *x, int rank);
        void destroy_matrix();
        void release_ranks();
    };

} // namespace polysolve::linear
