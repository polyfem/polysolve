#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

#include <cudss.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>

#include <set>

extern "C"
{
    HYPRE_Int hypre_ParVectorAxpy(HYPRE_Complex alpha, HYPRE_ParVector x, HYPRE_ParVector y);
}

namespace polysolve::linear
{

    class GPUHybridSolver : public Solver
    {

    public:
        GPUHybridSolver();
        ~GPUHybridSolver();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(GPUHybridSolver)

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

        void check_settings() const;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override
        {
            return select_bad_dofs_from_l1_row_norm ? "GPUHybrid" : "GPUAMGF";
        }

        virtual void set_problematic_dofs(const std::set<int> &bad_dofs) override { h_all_bad_dofs = bad_dofs; }

        // Alternative strategies for picking which DOFs get the dense/direct
        // subdomain treatment, in place of the default GMM row-norm clustering.
        enum class SubdomainSelectionStrategy
        {
            KNEE,
            GMM,
            FD,
            COST,
            APOSTERIORI
        };

    protected:
        // AMG settings
        double theta = 0.5;

        // Hybrid preconditioner settings
        // When false, bad DOFs are taken only from whatever set_problematic_dofs
        // supplies instead of being inferred from the row-norm heuristic below
        // (AMGF mode).
        bool select_bad_dofs_from_l1_row_norm = true;
        SubdomainSelectionStrategy subdomain_selection_strategy = SubdomainSelectionStrategy::GMM;
        bool decompose_subdomains = true;
        int min_subdomain_size = 1;
        int max_subdomain_size = 1e9;
        double gmm_jump_threshold = 10.0;
        double gmm_tol = 1e-3;
        int max_gmm_iterations = 20;
        bool expand_subdomains = true;
        bool additive_mode = false;
        // Give each contact patch its own (possibly overlapping) subdomain,
        // solved additively -- an additive Schwarz treatment of contact DOFs.
        // Unlike the generic (row-norm/GMM-selected) subdomains, contact
        // patches are factorized and solved with a fixed-size dense batched
        // GPU solve rather than the sparse batched cuDSS path -- see
        // factorize_contact_patches_dense().
        bool contact_patch_schwarz = false;

        // General solver settings
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)

        // Optional per-row function (block) assignment for HYPRE_BoomerAMGSetDofFunc.
        // Empty means the default interleaved mapping (row i belongs to function i % dimension_).
        Eigen::VectorXi block_mapping_;
        int max_iter_ = 10000;
        double rel_conv_tol_ = 1e-10;
        double abs_conv_tol_ = 0.0;
        double conditioning_threshold = 100.0;

        // When false, skips the per-PCG-iteration timing logs (matmul,
        // amg_v_cycle, subdomain_solve, contact_patch_dense_subdomain_solve,
        // pcg_iter) so a long solve doesn't flood the log; every other log
        // line is unaffected.
        bool detailed_log = true;

        // solve information
        HYPRE_Int num_iterations;
        HYPRE_Complex final_res_norm;

    private:
        bool has_matrix_ = false;

        // Hypre variables
        HYPRE_IJMatrix A;
        HYPRE_ParCSRMatrix parcsr_A;
        HYPRE_IJVector ij_x;
        HYPRE_IJVector ij_b;

        // hybrid preconditioner data
        std::set<int> h_all_bad_dofs;
        std::vector<int> h_subdomain_sizes;
        thrust::device_vector<int> d_subdomain_sizes;
        thrust::device_vector<int> d_all_bad_dofs;
        std::vector<std::vector<int>> bad_indices_arrays;

        thrust::device_vector<int> d_inner_indices;
        thrust::device_vector<int> d_outer_indices;
        thrust::device_vector<double> d_values;

        cudssHandle_t cudss_handle = nullptr;
        cudssConfig_t cudss_config = nullptr;
        cudssData_t cudss_solver_data = nullptr;
        cudssMatrix_t batch_A = nullptr;
        cudssMatrix_t batch_x = nullptr;
        cudssMatrix_t batch_b = nullptr;

        cublasHandle_t cublas_handle = nullptr;

        int sparse_batch_count = 0;

        thrust::device_vector<int> d_sparse_dof_map;

        // sparse solve data
        std::vector<int> h_sparse_nrows, h_sparse_ncols, h_sparse_nnz, h_sparse_vec_ncols, h_sparse_ld;

        thrust::device_vector<int> d_sparse_inner_indices, d_sparse_outer_indices;
        thrust::device_vector<double> d_sparse_values;
        thrust::device_vector<double> d_sparse_x, d_sparse_b;
        thrust::device_vector<void *> d_sparse_inner_void, d_sparse_outer_void, d_sparse_values_void;
        thrust::device_vector<void *> d_sparse_x_void, d_sparse_b_void;

        // contact_patch_schwarz: fixed-size dense batched solve data. Every
        // contact patch is padded (with an identity block on the unused
        // rows/cols) up to this many dofs and factorized/solved as one dense
        // block per patch via cublas<t>getrfBatched/getrsBatched -- see
        // factorize_contact_patches_dense().
        static constexpr int kContactPatchDenseSize = 64;
        int num_contact_patches_ = 0;
        int total_contact_patch_real_dofs_ = 0;

        thrust::device_vector<double> d_dense_A;      // num_contact_patches_ * 64 * 64, column-major, LU factors in place
        thrust::device_vector<double *> d_dense_A_ptrs; // num_contact_patches_ device pointers into d_dense_A
        thrust::device_vector<int> d_dense_pivot;     // num_contact_patches_ * 64
        thrust::device_vector<int> d_dense_info;      // num_contact_patches_ (getrfBatched status, device)

        thrust::device_vector<double> d_dense_b;      // num_contact_patches_ * 64, padded rhs/solution
        thrust::device_vector<double *> d_dense_b_ptrs; // num_contact_patches_ device pointers into d_dense_b

        // Per real (non-padding) dof, patch-major: its global dof index, and
        // its flat slot (patch * 64 + local_index) in the padded layout above.
        thrust::device_vector<int> d_contact_dof_map;
        thrust::device_vector<int> d_contact_real_to_dense_slot;

        // Contact patches may overlap, so a dof's correction is the sum of
        // every patch that covers it. This global (not per-dof) scale factor
        // -- 1 / (average number of patches covering a covered dof) -- damps
        // that sum so overlapping patches don't over-correct on average.
        double overlap_scale_ = 1.0;
        int num_unique_overlap_dofs_ = 0;

        // Scratch for summing overlapping patches' contributions to a shared
        // dof without atomics: sort real entries by global dof once at
        // factorize time (d_reduce_perm/d_reduce_keys), then each iteration
        // gather solved values into that order and thrust::reduce_by_key them.
        thrust::device_vector<int> d_reduce_perm;
        thrust::device_vector<int> d_reduce_keys;
        thrust::device_vector<double> d_reduce_values;
        thrust::device_vector<int> d_reduce_unique_keys;
        thrust::device_vector<double> d_reduce_unique_values;

    public:
        void free_device_memory();

        // factorization helpers
        void copy_matrix_to_hypre();

        // solve helpers
        void init_hypre_vectors(const int size);

        // hybrid preconditioner helpers
        void decompose_subdomains_to_disjoint_subsets(const Eigen::SparseMatrix<double> &sparse_A);
        void filter_subdomains(const Eigen::SparseMatrix<double> &sparse_A);
        void expand_subdomains_to_strongly_connected(const Eigen::SparseMatrix<double> &sparse_A);
        void select_bad_dofs();
        void factorize_submatrix();
        void factorize_contact_patches_dense(const Eigen::SparseMatrix<double> &sparse_A);
        void compute_overlap_scale();
        void free_contact_patch_dense_memory();

        // linear algebra
        void set_hypre_vec(HYPRE_IJVector &ij_x, HYPRE_ParVector &par_x, const thrust::device_vector<double> &x);
        void matmul(const thrust::device_vector<double> &x, thrust::device_vector<double> &result);
        double dot(const thrust::device_vector<double> &x, const thrust::device_vector<double> &y);
        void vector_copy(const thrust::device_vector<double> &x, thrust::device_vector<double> &y);
        void vector_add(double alpha, const thrust::device_vector<double> &x, thrust::device_vector<double> &y);
        void vector_scale(double alpha, thrust::device_vector<double> &x);

        // preconditioning functions
        void custom_mixed_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double> &r, thrust::device_vector<double> &z, thrust::device_vector<double> &buffer, thrust::device_vector<double> &z2);
        void amg_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double> &b, thrust::device_vector<double> &x);
        void dss_precond_iter(thrust::device_vector<double> &z, thrust::device_vector<double> &r, thrust::device_vector<double> &next_z);
        void dense_contact_patch_precond_iter(thrust::device_vector<double> &z, thrust::device_vector<double> &r, thrust::device_vector<double> &next_z);

        // Krylov solve methods
        void pcg_solve(thrust::device_vector<double> &rhs, thrust::device_vector<double> &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond);
    };

} // namespace polysolve::linear