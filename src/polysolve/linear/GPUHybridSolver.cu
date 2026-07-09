

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

#include <iostream>
#include <fstream>

#include <chrono>
#include <stdexcept>

#include <spdlog/spdlog.h>

#ifdef HYPRE_ENABLE_MPI
#include <mpi.h>
#endif


#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorName(status) \
                      << " (" << cudaGetErrorString(status) << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUDSS(call) \
    do { \
        cudssStatus_t status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            std::cerr << "cuDSS Error at " << __FILE__ << ":" << __LINE__ \
                      << " code " << (int) status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


namespace polysolve::linear
{

    namespace
    {
        using clock = std::chrono::steady_clock;

        double elapsed_seconds(const std::chrono::time_point<clock> &begin)
        {
            return std::chrono::duration<double>(clock::now() - begin).count();
        }
    }

    GPUHybridSolver::GPUHybridSolver()
    {        
#ifdef HYPRE_ENABLE_MPI
        int done_already;

        MPI_Initialized(&done_already);
        if (!done_already)
        {
            /* Initialize MPI */
            int argc = 1;
            char name[] = "";
            char *argv[] = {name};
            char **argvv = &argv[0];
            MPI_Init(&argc, &argvv);
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
            if (params["GPUHybrid"].contains("block_dim"))
            {
                dimension_ = params["GPUHybrid"]["block_dim"];
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
        }
    } 

    void GPUHybridSolver::get_info(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
    }

    void GPUHybridSolver::check_settings() const
    {
        
    }

    void GPUHybridSolver::factorize(const StiffnessMatrix &Ain)
    {
        check_settings();
        SPDLOG_TRACE("[{}] [start_solve] [0.000000] [problem_size={}]", name(), Ain.rows());

        {
            auto phase_begin = clock::now();

            d_outer_indices.resize(Ain.rows() + 1);
            d_inner_indices.resize(Ain.nonZeros());
            d_values.resize(Ain.nonZeros());

            thrust::copy(Ain.outerIndexPtr(), Ain.outerIndexPtr() + d_outer_indices.size(), d_outer_indices.begin());
            thrust::copy(Ain.innerIndexPtr(), Ain.innerIndexPtr() + d_inner_indices.size(), d_inner_indices.begin());
            thrust::copy(Ain.valuePtr(), Ain.valuePtr() + d_values.size(), d_values.begin());

            SPDLOG_TRACE("[{}] [copy_matrix_to_gpu] [{:.6f}]", name(), elapsed_seconds(phase_begin));
        }

        {
            auto phase_begin = clock::now();
            
            bad_indices_arrays.clear();
            select_bad_dofs();

            if (decompose_subdomains)
            {
                filter_subdomains(Ain);
            }
            
            if (expand_subdomains)
            {
                expand_subdomains_to_strongly_connected(Ain);
            }

            if (decompose_subdomains)
            {
                decompose_subdomains_to_disjoint_subsets(Ain);
            }
            else 
            {
                bad_indices_arrays.emplace_back(h_all_bad_dofs.begin(), h_all_bad_dofs.end());
            }

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

            SPDLOG_TRACE("[{}] [setup_problematic_dof_precond] [{:.6f}]", name(), elapsed_seconds(phase_begin));
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

    namespace {
        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 8; // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
            int agg_levels = 1;    // number of aggressive coarsening levels
            double theta = 0.25;   // strength threshold: 0.25, 0.5, 0.8

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i, 0 = classical
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 18;   // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
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
            //print_level = 3;
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

            //HYPRE_BoomerAMGSetDofFunc(amg_precond, (HYPRE_Int*) dof_to_function.data());

            // More robust options with respect to convergence
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
        }
    }

    void GPUHybridSolver::solve(const Ref<const VectorXd> b, Ref<VectorXd> x)
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
                    theta
                );
            }

            HYPRE_BoomerAMGSetup(precond, parcsr_A, par_b, par_x);
            CHECK_CUDA(cudaDeviceSynchronize());
            SPDLOG_TRACE("[{}] [amg_setup] [{:.6f}]", name(), elapsed_seconds(phase_begin));
        }

        {
            auto phase_begin = clock::now();

            pcg_solve(d_b, d_x, par_b, par_x, precond);

            thrust::device_vector<double> buffer(d_x.size());
            matmul(d_x, buffer);
            vector_add(-1.0, d_b, buffer);
            final_res_norm = sqrt(dot(buffer, buffer));

            thrust::copy(d_x.begin(), d_x.end(), x.data());

            CHECK_CUDA(cudaDeviceSynchronize());
            SPDLOG_TRACE("[{}] [pcg_solve] [{:.6f}] [pcg_iters={}] [residual={}]", name(), elapsed_seconds(phase_begin), num_iterations, final_res_norm);
        }

        {
            HYPRE_BoomerAMGDestroy(precond);
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
        const HYPRE_Int* raw_outer = thrust::raw_pointer_cast(d_outer_indices.data());
        HYPRE_Int* raw_n_cols = thrust::raw_pointer_cast(d_n_cols.data());

        thrust::for_each(thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_rows),
            [=] __device__ (int i) {
                raw_n_cols[i] = raw_outer[i + 1] - raw_outer[i];
            }
        );

        HYPRE_Int* gpu_n_cols = thrust::raw_pointer_cast(d_n_cols.data());
        HYPRE_Int* gpu_rows   = thrust::raw_pointer_cast(d_rows.data());
        
        HYPRE_Int* gpu_cols   = thrust::raw_pointer_cast(d_inner_indices.data()); 
        double* gpu_vals   = thrust::raw_pointer_cast(d_values.data());

        HYPRE_IJMatrixSetValues(A, num_rows, gpu_n_cols, gpu_rows, gpu_cols, gpu_vals);

        HYPRE_IJMatrixAssemble(A);

        void* temp_A = nullptr;
        HYPRE_IJMatrixGetObject(A, &temp_A);
        parcsr_A = static_cast<decltype(parcsr_A)>(temp_A);

        SPDLOG_TRACE("[{}] [copy_matrix_to_hypre] [{:.6f}]", name(), elapsed_seconds(phase_begin));
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

    void GPUHybridSolver::matmul(const thrust::device_vector<double>& x, thrust::device_vector<double>& result)
    {
        auto phase_begin = clock::now();        
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_result;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_result, result);

        HYPRE_ParCSRMatrixMatvec(1.0, parcsr_A, par_x, 0.0, par_result);
        CHECK_CUDA(cudaDeviceSynchronize());
        SPDLOG_TRACE("[{}] [matmul] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    double GPUHybridSolver::dot(const thrust::device_vector<double>& a, const thrust::device_vector<double>& b)
    {
        HYPRE_ParVector par_a;
        HYPRE_ParVector par_b;

        set_hypre_vec(ij_x, par_a, a);
        set_hypre_vec(ij_b, par_b, b);
        
        double result;
        HYPRE_ParVectorInnerProd(par_a, par_b, &result);
        return result;
    }

    void GPUHybridSolver::vector_copy(const thrust::device_vector<double>& x, thrust::device_vector<double>& y)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_y;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_y, y);

        HYPRE_ParVectorCopy(par_x, par_y);
    }

    void GPUHybridSolver::vector_add(double alpha, const thrust::device_vector<double>& x, thrust::device_vector<double>& y)
    {
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_y;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_y, y);

        hypre_ParVectorAxpy(alpha, par_x, par_y);
    }

    void GPUHybridSolver::vector_scale(double alpha, thrust::device_vector<double>& x)
    {
        HYPRE_ParVector par_x;

        set_hypre_vec(ij_x, par_x, x);

        HYPRE_ParVectorScale(alpha, par_x);
    }

    void GPUHybridSolver::set_hypre_vec(HYPRE_IJVector &my_ij_x, HYPRE_ParVector &par_x, const thrust::device_vector<double>& x)
    {
        double* raw_ptr = const_cast<double*>(thrust::raw_pointer_cast(x.data()));
        
        HYPRE_IJVectorSetData(my_ij_x, raw_ptr);
        HYPRE_IJVectorAssemble(my_ij_x);
        HYPRE_IJVectorGetObject(my_ij_x, (void **)&par_x);
    }

    void GPUHybridSolver::custom_mixed_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double>& r, thrust::device_vector<double>& z, thrust::device_vector<double>& buffer, thrust::device_vector<double>& z2)
    {
        if (d_all_bad_dofs.size() == 0)
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

    void GPUHybridSolver::dss_precond_iter(thrust::device_vector<double>& z, thrust::device_vector<double>& r, thrust::device_vector<double>& next_z)
    {
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
                d_sparse_b.begin()
            );
            
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
                next_z.begin()
            );
        }

        vector_add(1.0, z, next_z);

        CHECK_CUDA(cudaDeviceSynchronize());
        SPDLOG_TRACE("[{}] [subdomain_solve] [{:.6f}]", name(), elapsed_seconds(phase_begin));
    }

    void GPUHybridSolver::amg_precond_iter(const HYPRE_Solver &precond, thrust::device_vector<double>& b, thrust::device_vector<double>& x)
    {
        auto phase_begin = clock::now();
        HYPRE_ParVector par_x;
        HYPRE_ParVector par_b;

        set_hypre_vec(ij_x, par_x, x);
        set_hypre_vec(ij_b, par_b, b);

        HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);
        CHECK_CUDA(cudaDeviceSynchronize());
        SPDLOG_TRACE("[{}] [amg_v_cycle] [{:.6f}]", name(), elapsed_seconds(phase_begin));
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

        SPDLOG_TRACE("[{}] [subdomain_decomposition] [{}] [num_subdomains={}] ", \
            name(), elapsed_seconds(phase_begin), bad_indices_arrays.size());
    }

    void GPUHybridSolver::select_bad_dofs()
    {
        auto phase_begin = clock::now();

        const int num_rows = d_outer_indices.size() - 1;

        const int* row_offsets = thrust::raw_pointer_cast(d_outer_indices.data());
        const double* values   = thrust::raw_pointer_cast(d_values.data());

        thrust::device_vector<double> d_row_norms(num_rows);
        double* row_norms = thrust::raw_pointer_cast(d_row_norms.data());

        thrust::for_each(thrust::device,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_rows),
            [=] __device__ (int i) {
                int start = row_offsets[i];
                int end = row_offsets[i + 1];
                double sum = 0.0;
                for (int j = start; j < end; ++j) {
                    sum += fabs(values[j]);
                }
                row_norms[i] = sum;
            }
        );

        double sum_norms = thrust::reduce(d_row_norms.begin(), d_row_norms.end(), 0.0);
        double global_mean = sum_norms / num_rows;

        double var_sum = thrust::transform_reduce(thrust::device,
            d_row_norms.begin(), d_row_norms.end(), 
            [global_mean] __device__ (double x) -> double { return (x - global_mean) * (x - global_mean); }, 
            0.0, thrust::plus<double>()
        );
        double global_var = var_sum / num_rows;

        auto minmax = thrust::minmax_element(d_row_norms.begin(), d_row_norms.end());
        double mean_0 = *minmax.first;
        double mean_1 = *minmax.second;
        double var_0 = global_var;
        double var_1 = global_var;
        double w0 = 0.5;
        double w1 = 0.5;

        double var_reg = 1e-6;

        thrust::device_vector<double> d_gamma0(num_rows);
        thrust::device_vector<double> d_gamma1(num_rows);
        double* g0 = thrust::raw_pointer_cast(d_gamma0.data());
        double* g1 = thrust::raw_pointer_cast(d_gamma1.data());

        int gmm_iter;

        for (gmm_iter = 0; gmm_iter < max_gmm_iterations; ++gmm_iter) {
            
            double log_likelihood = thrust::transform_reduce(thrust::device,
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(num_rows),
                [=] __device__ (int i) -> double {
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

                    return log_total; 
                },
                0.0, thrust::plus<double>()
            );

            double sum_g0 = thrust::reduce(d_gamma0.begin(), d_gamma0.end(), 0.0);
            double sum_g1 = thrust::reduce(d_gamma1.begin(), d_gamma1.end(), 0.0);

            w0 = sum_g0 / num_rows;
            w1 = sum_g1 / num_rows;

            double old_mean_0 = mean_0, old_mean_1 = mean_1;
            double old_var_0 = var_0, old_var_1 = var_1;

            mean_0 = thrust::inner_product(d_gamma0.begin(), d_gamma0.end(), d_row_norms.begin(), 0.0) / sum_g0;
            mean_1 = thrust::inner_product(d_gamma1.begin(), d_gamma1.end(), d_row_norms.begin(), 0.0) / sum_g1;

            var_0 = thrust::transform_reduce(thrust::device,
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(num_rows),
                [=] __device__ (int i) -> double { return g0[i] * (row_norms[i] - mean_0) * (row_norms[i] - mean_0); },
                0.0, thrust::plus<double>()
            ) / sum_g0 + var_reg;

            var_1 = thrust::transform_reduce(thrust::device,
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(num_rows),
                [=] __device__ (int i) -> double { return g1[i] * (row_norms[i] - mean_1) * (row_norms[i] - mean_1); },
                0.0, thrust::plus<double>()
            ) / sum_g1 + var_reg;

            // Check Convergence
            if (abs(mean_0 - old_mean_0) / abs(old_mean_0) < gmm_tol && 
                abs(mean_1 - old_mean_1) / abs(old_mean_1) < gmm_tol && 
                abs(var_0 - old_var_0) / abs(old_var_0) < gmm_tol && 
                abs(var_1 - old_var_1) / abs(old_var_1) < gmm_tol) 
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
                [=] __device__ (int i) { return g0[i] < g1[i]; }
            );

            num_bad_dofs = thrust::distance(d_all_bad_dofs.begin(), end_it);
        }

        d_all_bad_dofs.resize(num_bad_dofs);

        std::vector<int> h_bad_dofs(num_bad_dofs);
        thrust::copy(d_all_bad_dofs.begin(), d_all_bad_dofs.end(), h_bad_dofs.begin());

        h_all_bad_dofs.clear();
        h_all_bad_dofs.insert(h_bad_dofs.begin(), h_bad_dofs.end());

        SPDLOG_TRACE("[{}] [bad_dof_selection] [{:.6f}] [global_mean={}] [global_var={}] [mean_0={}] [mean_1={}] [var_0={}] [var_1={}] [gmm_iters={}] [num_bad_dofs={}]", 
            name(), elapsed_seconds(phase_begin), global_mean, global_var, mean_0, mean_1, var_0, var_1, gmm_iter, num_bad_dofs);
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

        SPDLOG_TRACE("[{}] [subdomain_filtering] [{}] [total_dofs_before={}] [total_dofs_after={}] [num_too_small={}] [num_too_large={}] [num_not_poorly_conditioned={}]", \
            name(), elapsed_seconds(phase_begin), original_num_bad_dofs, h_all_bad_dofs.size(), num_too_small, num_too_large, num_not_poorly_conditioned);
    }

    void GPUHybridSolver::expand_subdomains_to_strongly_connected(const Eigen::SparseMatrix<double> &sparse_A)
    {
        auto phase_begin = clock::now();
        int num_bad_dofs_before = h_all_bad_dofs.size();

        std::set<int> new_bad_dofs;;

        for (int k : h_all_bad_dofs)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_A, k); it; ++it)
            {
                new_bad_dofs.insert(it.row());
            }
        }

        h_all_bad_dofs = std::move(new_bad_dofs);

        SPDLOG_TRACE("[{}] [subdomain_expansion] [{}] [num_dofs_before={}] [num_dofs_after={}]", \
            name(), elapsed_seconds(phase_begin), num_bad_dofs_before, h_all_bad_dofs.size());
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
            
            int* raw_bad_dofs = thrust::raw_pointer_cast(d_sparse_dof_map.data());
            int* raw_batch_offsets = thrust::raw_pointer_cast(d_sparse_batch_offsets.data());
            int* raw_batch_sizes = thrust::raw_pointer_cast(d_sparse_batch_sizes.data());
            int* raw_outer = thrust::raw_pointer_cast(d_outer_indices.data());
            int* raw_inner = thrust::raw_pointer_cast(d_inner_indices.data());
            
            thrust::device_vector<int> d_row_nnz(total_sparse_dofs, 0);
            int* raw_row_nnz = thrust::raw_pointer_cast(d_row_nnz.data());
            
            thrust::fill(d_sparse_nnz.begin(), d_sparse_nnz.end(), 0);
            int* raw_sparse_nnz = thrust::raw_pointer_cast(d_sparse_nnz.data());

            int local_sparse_batch_count = sparse_batch_count;

            thrust::for_each(thrust::device,
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(total_sparse_dofs),
                [=] __device__ (int i) {
                    
                    int* batch_ptr = thrust::upper_bound(thrust::seq, raw_batch_offsets, raw_batch_offsets + local_sparse_batch_count, i);
                    int batch_idx = (batch_ptr - raw_batch_offsets) - 1;
                    
                    int offset = raw_batch_offsets[batch_idx];
                    int size = raw_batch_sizes[batch_idx];
                    
                    int global_row = raw_bad_dofs[i];
                    int row_start = raw_outer[global_row];
                    int row_end = raw_outer[global_row + 1];
                    
                    int* sub_begin = raw_bad_dofs + offset;
                    int* sub_end = sub_begin + size;
                    
                    int row_nnz = 0;
                    for (int j = row_start; j < row_end; ++j) {
                        int global_col = raw_inner[j];
                        if (thrust::binary_search(thrust::seq, sub_begin, sub_end, global_col)) {
                            row_nnz++;
                        }
                    }
                    
                    raw_row_nnz[i] = row_nnz;
                    atomicAdd(&raw_sparse_nnz[batch_idx], row_nnz);
                }
            );

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
            
            int* raw_row_nnz_starts = thrust::raw_pointer_cast(d_row_nnz_starts.data());
            int* raw_sparse_outer = thrust::raw_pointer_cast(d_sparse_outer_indices.data());
            int* raw_sparse_inner = thrust::raw_pointer_cast(d_sparse_inner_indices.data());
            
            double* raw_global_values = thrust::raw_pointer_cast(d_values.data());
            double* raw_sparse_values = thrust::raw_pointer_cast(d_sparse_values.data());

            thrust::for_each(thrust::device,
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(total_sparse_dofs),
                [=] __device__ (int i) {
                    
                    int* batch_ptr = thrust::upper_bound(thrust::seq, raw_batch_offsets, raw_batch_offsets + local_sparse_batch_count, i);
                    int batch_idx = (batch_ptr - raw_batch_offsets) - 1;
                    
                    int offset = raw_batch_offsets[batch_idx];
                    int size = raw_batch_sizes[batch_idx];
                    
                    int r = i - offset; 
                    int outer_start = offset + batch_idx; 
                    
                    if (r == 0) {
                        raw_sparse_outer[outer_start] = 0; 
                    }
                    
                    int batch_nnz_start = raw_row_nnz_starts[offset];
                    
                    raw_sparse_outer[outer_start + r + 1] = (raw_row_nnz_starts[i] + raw_row_nnz[i]) - batch_nnz_start;

                    int current_nnz = raw_row_nnz_starts[i];
                    
                    int global_row = raw_bad_dofs[i];
                    int row_start = raw_outer[global_row];
                    int row_end = raw_outer[global_row + 1];
                    
                    int* sub_begin = raw_bad_dofs + offset;
                    int* sub_end = sub_begin + size;
                    
                    for (int j = row_start; j < row_end; ++j) {
                        int global_col = raw_inner[j];
                        
                        int* ptr = thrust::lower_bound(thrust::seq, sub_begin, sub_end, global_col);
                        
                        if (ptr != sub_end && *ptr == global_col) {
                            int local_col = ptr - sub_begin;
                            
                            raw_sparse_inner[current_nnz] = local_col;
                            raw_sparse_values[current_nnz] = raw_global_values[j];
                            
                            current_nnz++;
                        }
                    }
                }
            );

            std::vector<void*> h_sparse_outer_void;
            std::vector<void*> h_sparse_inner_void;
            std::vector<void*> h_sparse_values_void;
            std::vector<void*> h_sparse_x_void;
            std::vector<void*> h_sparse_b_void;

            for (int i = 0; i < sparse_batch_count; ++i)
            {
                h_sparse_outer_void.push_back(static_cast<void*>(thrust::raw_pointer_cast(d_sparse_outer_indices.data()) + h_sparse_row_starts[i]));
                h_sparse_inner_void.push_back(static_cast<void*>(thrust::raw_pointer_cast(d_sparse_inner_indices.data()) + h_sparse_nnz_starts[i]));
                h_sparse_x_void.push_back(static_cast<void*>(thrust::raw_pointer_cast(d_sparse_x.data()) + h_sparse_dof_starts[i]));
                h_sparse_b_void.push_back(static_cast<void*>(thrust::raw_pointer_cast(d_sparse_b.data()) + h_sparse_dof_starts[i]));
                h_sparse_values_void.push_back(static_cast<void*>(thrust::raw_pointer_cast(d_sparse_values.data()) + h_sparse_nnz_starts[i]));
                
            }

            d_sparse_outer_void = h_sparse_outer_void;
            d_sparse_inner_void = h_sparse_inner_void;
            d_sparse_values_void = h_sparse_values_void;
            d_sparse_x_void = h_sparse_x_void;
            d_sparse_b_void = h_sparse_b_void;

            auto precision = CUDA_R_64F;

            CHECK_CUDSS(cudssMatrixCreateBatchDn(
                &batch_x, sparse_batch_count, h_sparse_nrows.data(), h_sparse_vec_ncols.data(), h_sparse_ld.data(), 
                thrust::raw_pointer_cast(d_sparse_x_void.data()), CUDA_R_32I, precision, CUDSS_LAYOUT_COL_MAJOR
            ));

            CHECK_CUDSS(cudssMatrixCreateBatchDn(
                &batch_b, sparse_batch_count, h_sparse_nrows.data(), h_sparse_vec_ncols.data(), h_sparse_ld.data(), 
                thrust::raw_pointer_cast(d_sparse_b_void.data()), CUDA_R_32I, precision, CUDSS_LAYOUT_COL_MAJOR
            ));

            CHECK_CUDSS(cudssMatrixCreateBatchCsr(
                &batch_A, sparse_batch_count, h_sparse_nrows.data(), h_sparse_ncols.data(), h_sparse_nnz.data(), 
                thrust::raw_pointer_cast(d_sparse_outer_void.data()), nullptr, thrust::raw_pointer_cast(d_sparse_inner_void.data()), thrust::raw_pointer_cast(d_sparse_values_void.data()), 
                CUDA_R_32I, precision, CUDSS_MTYPE_SYMMETRIC, 
                CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO
            ));

            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_ANALYSIS, cudss_config, cudss_solver_data, batch_A, nullptr, nullptr));
            CHECK_CUDSS(cudssExecute(cudss_handle, CUDSS_PHASE_FACTORIZATION, cudss_config, cudss_solver_data, batch_A, nullptr, nullptr));

        }

        CHECK_CUDA(cudaDeviceSynchronize());
        SPDLOG_TRACE("[{}] [factorize_submatrix] [{}] [n_sparse={}] [n_sparse_dofs={}]", \
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
                return;
            }

            matmul(result, buffer);
            
            vector_copy(rhs, r);
            vector_add(-1.0, buffer, r);

            custom_mixed_precond_iter(precond, r, z, buffer, z2);

            vector_copy(z, p);

            gamma = dot(r, z);
            old_gamma = gamma;
            SPDLOG_TRACE("[{}] [pre_loop] [{:.6f}] [rhs_norm={}]", name(), elapsed_seconds(phase_begin), sqrt(bi_prod));
        }

        for (int k = 0; k < max_iter_; ++k)
        {
            auto phase_begin = clock::now();
            num_iterations = k + 1;

            matmul(p, buffer);
            double sdotp = dot(p, buffer);

            if (sdotp == 0.0)
            {
                SPDLOG_TRACE("[{}] [err_zero_sdotp] [0.000000]", name());
                break;
            }

            double alpha = gamma / sdotp;

            if (alpha <= 0.0)
            {
                SPDLOG_TRACE("[{}] [err_negative_alpha] [0.000000]", name());
                break;
            } 
            else if (alpha < __DBL_MIN__)
            {
                SPDLOG_TRACE("[{}] [err_subnormal_alpha] [0.000000]", name());
                break;
            }

            vector_add(alpha, p, result);
            vector_add(-1.0 * alpha, buffer, r);
            double i_prod = dot(r, r);
            
            if (rel_eps > 0 && (i_prod / bi_prod) < rel_eps)
            {
                SPDLOG_TRACE("[{}] [converged_rel] [0.000000]", name());
                break;
            }

            if (abs_eps > 0 && i_prod < abs_eps)
            {
                SPDLOG_TRACE("[{}] [converged_abs] [0.000000]", name());
                break;
            }

            thrust::fill(z.begin(), z.end(), 0.0);
            custom_mixed_precond_iter(precond, r, z, buffer, z2);

            gamma = dot(r, z);
            
            double beta = gamma / old_gamma;
            old_gamma = gamma;

            vector_scale(beta, p);
            vector_add(1.0, z, p);

            CHECK_CUDA(cudaDeviceSynchronize());
            SPDLOG_TRACE("[{}] [pcg_iter] [{:.6f}] [iter={}] [residual={}]", name(), elapsed_seconds(phase_begin), k, sqrt(i_prod));
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

        if (cudss_handle) {
            cudssDestroy(cudss_handle);
            cudss_handle = nullptr;
        }
    }

    void GPUHybridSolver::free_device_memory() 
    {
        // Destroy cuDSS Opaque Structures
        if (batch_A) { CHECK_CUDSS(cudssMatrixDestroy(batch_A)); batch_A = nullptr; }
        if (batch_x) { CHECK_CUDSS(cudssMatrixDestroy(batch_x)); batch_x = nullptr; }
        if (batch_b) { CHECK_CUDSS(cudssMatrixDestroy(batch_b)); batch_b = nullptr; }
        if (cudss_solver_data)   { CHECK_CUDSS(cudssDataDestroy(cudss_handle, cudss_solver_data)); cudss_solver_data = nullptr; }
        if (cudss_config)       { CHECK_CUDSS(cudssConfigDestroy(cudss_config)); cudss_config = nullptr; }
    }
}