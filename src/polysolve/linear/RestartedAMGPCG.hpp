#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

#include <thrust/device_vector.h>

extern "C"
{
    HYPRE_Int hypre_ParVectorAxpy(HYPRE_Complex alpha, HYPRE_ParVector x, HYPRE_ParVector y);
}

namespace polysolve::linear
{

    // BoomerAMG-preconditioned CG on the GPU. AMG is set up exactly as in
    // GPUHybridSolver (minus its problematic-dof subspace correction). Every
    // restart_interval iterations the true residual b - Ax is recomputed in
    // double-double precision and CG is restarted from it, so the
    // recursively-updated residual can't drift away from the true one;
    // convergence is only ever declared on that recomputed residual.
    class RestartedAMGPCG : public Solver
    {

    public:
        RestartedAMGPCG();
        ~RestartedAMGPCG();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(RestartedAMGPCG)

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

        // Retrieve solve information
        virtual void get_info(json &params) const override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "RestartedAMGPCG"; }

    protected:
        // AMG settings
        double theta = 0.5;

        // Number of PCG iterations between double-double residual
        // recomputations (each of which restarts PCG).
        int restart_interval = 50;

        // General solver settings
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)

        // Optional per-row function (block) assignment for HYPRE_BoomerAMGSetDofFunc.
        // Empty means the default interleaved mapping (row i belongs to function i % dimension_).
        Eigen::VectorXi block_mapping_;
        int max_iter_ = 10000;
        double rel_conv_tol_ = 1e-10;
        double abs_conv_tol_ = 0.0;

        // When false, skips the per-PCG-iteration logs (matmul, amg_v_cycle,
        // pcg_iter) so a long solve doesn't flood the log; every other log
        // line, including the per-restart one, is unaffected.
        bool detailed_log = true;

        // solve information
        int num_iterations = 0;
        int num_restarts = 0;
        double final_res_norm = 0.0;

    private:
        bool has_matrix_ = false;

        // Hypre variables
        HYPRE_IJMatrix A;
        HYPRE_ParCSRMatrix parcsr_A;
        HYPRE_IJVector ij_x;
        HYPRE_IJVector ij_b;

        // Device copy of the system matrix (CSR, same layout handed to
        // HYPRE). Unlike GPUHybridSolver, this is kept after factorize() for
        // the double-double residual, at the cost of a second copy of the
        // matrix alongside HYPRE's own.
        thrust::device_vector<int> d_inner_indices;
        thrust::device_vector<int> d_outer_indices;
        thrust::device_vector<double> d_values;

    public:
        // factorization helpers
        void copy_matrix_to_hypre();

        // solve helpers
        void init_hypre_vectors(const int size);

        // linear algebra
        void set_hypre_vec(HYPRE_IJVector &ij_x, HYPRE_ParVector &par_x, const thrust::device_vector<double> &x);
        void matmul(const thrust::device_vector<double> &x, thrust::device_vector<double> &result);
        double dot(const thrust::device_vector<double> &x, const thrust::device_vector<double> &y);
        void vector_copy(const thrust::device_vector<double> &x, thrust::device_vector<double> &y);
        void vector_add(double alpha, const thrust::device_vector<double> &x, thrust::device_vector<double> &y);
        void vector_scale(double alpha, thrust::device_vector<double> &x);

        // r = b - Ax, accumulated in double-double precision and rounded to double
        void dd_residual(const thrust::device_vector<double> &b, const thrust::device_vector<double> &x, thrust::device_vector<double> &r);

        // preconditioning functions
        void amg_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double> &b, thrust::device_vector<double> &x);

        // Krylov solve methods
        void pcg_solve(thrust::device_vector<double> &rhs, thrust::device_vector<double> &result, HYPRE_Solver &precond);
    };

} // namespace polysolve::linear
