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
#include <thrust/device_vector.h>

#include <set>

extern "C" {
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

        // Retrieve solve information
        virtual void get_info(json &params) const override;

        void check_settings() const;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { 
            return "GPUHybrid"; 
        }

    protected:
        // AMG settings
        double theta = 0.5;

        // Hybrid preconditioner settings
        bool decompose_subdomains = true;
        int min_subdomain_size = 1;
        int max_subdomain_size = 1e9;
        double gmm_jump_threshold = 10.0;
        double gmm_tol = 1e-3;
        int max_gmm_iterations = 20;
        bool expand_subdomains = true;
        bool additive_mode = false;

        // General solver settings
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 10000;
        double rel_conv_tol_ = 1e-10;
        double abs_conv_tol_ = 0.0;
        double conditioning_threshold = 100.0;

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

        int sparse_batch_count = 0;

        thrust::device_vector<int> d_sparse_dof_map;

        // sparse solve data
        std::vector<int> h_sparse_nrows, h_sparse_ncols, h_sparse_nnz, h_sparse_vec_ncols, h_sparse_ld;

        thrust::device_vector<int> d_sparse_inner_indices, d_sparse_outer_indices;
        thrust::device_vector<double> d_sparse_values;
        thrust::device_vector<double> d_sparse_x, d_sparse_b;
        thrust::device_vector<void*> d_sparse_inner_void, d_sparse_outer_void, d_sparse_values_void;
        thrust::device_vector<void*> d_sparse_x_void, d_sparse_b_void;

    public:
        void free_device_memory();

        // factorization helpers
        void copy_matrix_to_hypre();

        // solve helpers
        void init_hypre_vectors(const int size);

        // hybrid preconditioner helpers
        void decompose_subdomains_to_disjoint_subsets(const Eigen::SparseMatrix<double>& sparse_A);
        void filter_subdomains(const Eigen::SparseMatrix<double>& sparse_A);
        void expand_subdomains_to_strongly_connected(const Eigen::SparseMatrix<double>& sparse_A);
        void select_bad_dofs();
        void factorize_submatrix();

        // linear algebra
        void set_hypre_vec(HYPRE_IJVector &ij_x, HYPRE_ParVector &par_x, const thrust::device_vector<double>& x);
        void matmul(const thrust::device_vector<double>& x, thrust::device_vector<double>& result);
        double dot(const thrust::device_vector<double>& x, const thrust::device_vector<double>& y);
        void vector_copy(const thrust::device_vector<double>& x, thrust::device_vector<double>& y);
        void vector_add(double alpha, const thrust::device_vector<double>& x, thrust::device_vector<double>& y);
        void vector_scale(double alpha, thrust::device_vector<double>& x);

        // preconditioning functions
        void custom_mixed_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double> &r, thrust::device_vector<double> &z, thrust::device_vector<double> &buffer, thrust::device_vector<double> &z2);
        void amg_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double>& b, thrust::device_vector<double> &x);
        void dss_precond_iter(thrust::device_vector<double> &z, thrust::device_vector<double> &r, thrust::device_vector<double> &next_z);

        // Krylov solve methods
        void pcg_solve(thrust::device_vector<double> &rhs, thrust::device_vector<double> &result, HYPRE_ParVector &par_b, HYPRE_ParVector &par_x, HYPRE_Solver &precond);
    };

} // namespace polysolve::linear