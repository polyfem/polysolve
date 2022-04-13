#ifdef POLYSOLVE_WITH_AMGCL

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolverAMGCL.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{

    namespace
    {
        /* https://stackoverflow.com/questions/15904896/range-based-for-loop-on-a-dynamic-array */
        template <typename T>
        struct WrappedArray
        {
            WrappedArray(const T *first, const T *last)
                : begin_{first}, end_{last} {}
            WrappedArray(const T *first, std::ptrdiff_t size)
                : WrappedArray{first, first + size} {}

            const T *begin() const noexcept { return begin_; }
            const T *end() const noexcept { return end_; }
            const T &operator[](const size_t i) const { return begin_[i]; }

            const T *begin_;
            const T *end_;
        };
    } // namespace

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverAMGCL::LinearSolverAMGCL()
    {
        params_ = R"({
            "precond": {
                "relax": {
                    "type": "spai0"
                },
                "class": "amg",
                "direct_coarse": true,
                "ncycle": 1,
                "coarsening": {
                    "type": "smoothed_aggregation",
                    "estimate_spectral_radius": false,
                    "relax": 1.0
                }
            },
            "solver": {
                "tol": 1e-8,
                "maxiter": 1000,
                "type": "cg"
            }
        })"_json;

        // NOTE: usolver and psolver parameters are only used if the
        // preconditioner class is "schur_pressure_correction"
        precond_num_ = 0;
    }

    // Set solver parameters
    void LinearSolverAMGCL::setParameters(const json &params)
    {
        // Specially named parameters to match other solvers
        if (params.contains("max_iter"))
        {
            params_["solver"]["maxiter"] = params["max_iter"];
        }
        if (params.contains("conv_tol"))
        {
            params_["solver"]["tol"] = params["conv_tol"];
        }
        else if (params.contains("tolerance"))
        {
            params_["solver"]["tol"] = params["tolerance"];
        }
        if (params.contains("solver_type"))
        {
            params_["solver"]["type"] = params["solver_type"];
        }

        // Patch the stored params with input ones
        if (params.contains("precond"))
            params_["precond"].merge_patch(params["precond"]);
        if (params.contains("solver"))
            params_["solver"].merge_patch(params["solver"]);

        if (params_["precond"]["class"] == "schur_pressure_correction")
        {
            // Initialize the u and p solvers with a tolerance that is comparable to the main solver's
            if (!params_["precond"].contains("usolver"))
            {
                params_["precond"]["usolver"] = R"({"solver": {"maxiter": 100}})"_json;
                params_["precond"]["usolver"]["solver"]["tol"] = 10 * params_["solver"]["tol"].get<double>();
            }
            if (!params_["precond"].contains("usolver"))
            {
                params_["precond"]["psolver"] = R"({"solver": {"maxiter": 100}})"_json;
                params_["precond"]["psolver"]["solver"]["tol"] = 10 * params_["solver"]["tol"].get<double>();
            }
        }
    }

    void LinearSolverAMGCL::getInfo(json &params) const
    {
        params["num_iterations"] = iterations_;
        params["final_res_norm"] = residual_error_;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverAMGCL::factorize(const StiffnessMatrix &Ain)
    {
        assert(precond_num_ > 0);

        int numRows = Ain.rows();

        WrappedArray<StiffnessMatrix::StorageIndex> ia(Ain.outerIndexPtr(), numRows + 1);
        WrappedArray<StiffnessMatrix::StorageIndex> ja(Ain.innerIndexPtr(), Ain.nonZeros());
        WrappedArray<StiffnessMatrix::Scalar> a(Ain.valuePtr(), Ain.nonZeros());        
        if (params_["precond"]["class"] == "schur_pressure_correction")
        {
            std::vector<char> pmask(numRows, 0);
            for (size_t i = precond_num_; i < numRows; ++i)
                pmask[i] = 1;
            params_["precond"]["pmask"] = pmask;
        }
        
        // AMGCL takes the parameters as a Boost property_tree (i.e., another JSON data structure)
        std::stringstream ss_params;
        ss_params << params_;
        boost::property_tree::ptree pt_params;
        boost::property_tree::read_json(ss_params, pt_params);
        auto A=std::tie(numRows, ia, ja, a);
        solver_ = std::make_unique<Solver>(A, pt_params);
        std::cout<<*solver_<<std::endl;
        iterations_ = 0;
        residual_error_ = 0;
    }

    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverAMGCL::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        assert(result.size() == rhs.size());
        std::vector<double> _rhs(rhs.data(), rhs.data() + rhs.size());
        std::vector<double> x(result.data(), result.data() + result.size());
        auto rhs_b = Backend::copy_vector(_rhs, backend_params_);
        auto x_b = Backend::copy_vector(x, backend_params_);

        assert(solver_ != nullptr);
        std::tie(iterations_, residual_error_) = (*solver_)(*rhs_b, *x_b);

        std::copy(&(*x_b)[0], &(*x_b)[0] + result.size(), result.data());
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverAMGCL::~LinearSolverAMGCL()
    {
    }

    LinearSolverAMGCL_Block2::LinearSolverAMGCL_Block2()
    {
        params_ = R"({
            "precond": {
                "relax": {
                    "type": "spai0"
                },
                "class": "amg",
                "max_levels": 6,
                "direct_coarse": false,
                "ncycle": 2,
                "coarsening": {
                    "type": "smoothed_aggregation",
                    "estimate_spectral_radius": true,
                    "relax": 1,
                    "aggr": {
                        "eps_strong": 0
                    }
                }
            },
            "solver": {
                "tol": 1e-10,
                "maxiter": 1000,
                "type": "cg"
            }
        })"_json;

        // NOTE: usolver and psolver parameters are only used if the
        // preconditioner class is "schur_pressure_correction"
        precond_num_ = 0;
    }

    void LinearSolverAMGCL_Block2::setParameters(const json &params)
    {
        // Specially named parameters to match other solvers
        if (params.contains("max_iter"))
        {
            params_["solver"]["maxiter"] = params["max_iter"];
        }
        if (params.contains("conv_tol"))
        {
            params_["solver"]["tol"] = params["conv_tol"];
        }
        else if (params.contains("tolerance"))
        {
            params_["solver"]["tol"] = params["tolerance"];
        }
        if (params.contains("solver_type"))
        {
            params_["solver"]["type"] = params["solver_type"];
        }

        // Patch the stored params with input ones
        if (params.contains("precond"))
            params_["precond"].merge_patch(params["precond"]);
        if (params.contains("solver"))
            params_["solver"].merge_patch(params["solver"]);

        if (params_["precond"]["class"] == "schur_pressure_correction")
        {
            // Initialize the u and p solvers with a tolerance that is comparable to the main solver's
            if (!params_["precond"].contains("usolver"))
            {
                params_["precond"]["usolver"] = R"({"solver": {"maxiter": 100}})"_json;
                params_["precond"]["usolver"]["solver"]["tol"] = 10 * params_["solver"]["tol"].get<double>();
            }
            if (!params_["precond"].contains("usolver"))
            {
                params_["precond"]["psolver"] = R"({"solver": {"maxiter": 100}})"_json;
                params_["precond"]["psolver"]["solver"]["tol"] = 10 * params_["solver"]["tol"].get<double>();
            }
        }
    }

    void LinearSolverAMGCL_Block2::getInfo(json &params) const
    {
        params["num_iterations"] = iterations_;
        params["final_res_norm"] = residual_error_;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverAMGCL_Block2::factorize(const StiffnessMatrix &Ain)
    {
        assert(precond_num_ > 0);

        int numRows = Ain.rows();

        WrappedArray<StiffnessMatrix::StorageIndex> ia(Ain.outerIndexPtr(), numRows +1);
        WrappedArray<StiffnessMatrix::StorageIndex> ja(Ain.innerIndexPtr(), Ain.nonZeros());
        WrappedArray<StiffnessMatrix::Scalar> a(Ain.valuePtr(), Ain.nonZeros());

        if (params_["precond"]["class"] == "schur_pressure_correction")
        {
            std::vector<char> pmask(numRows, 0);
            for (size_t i = precond_num_; i < numRows; ++i)
                pmask[i] = 1;
            params_["precond"]["pmask"] = pmask;
        }

        // AMGCL takes the parameters as a Boost property_tree (i.e., another JSON data structure)
        std::stringstream ss_params;
        ss_params << params_;
        boost::property_tree::ptree pt_params;
        boost::property_tree::read_json(ss_params, pt_params);

        auto A = std::tie(numRows, ia, ja, a);
        auto Ab = amgcl::adapter::block_matrix<dmat_type>(A);
        solver_ = std::make_unique<Solver>(Ab, pt_params);

        iterations_ = 0;
        residual_error_ = 0;
    }

    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverAMGCL_Block2::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        assert(result.size() == rhs.size());
        std::vector<double> _rhs(rhs.data(), rhs.data() + rhs.size());
        std::vector<double> x(result.data(), result.data() + result.size());

        auto rhs_b=amgcl::backend::reinterpret_as_rhs<dmat_type>(_rhs);
        auto x_b = amgcl::backend::reinterpret_as_rhs<dmat_type>(x);

        assert(solver_ != nullptr);
        std::tie(iterations_, residual_error_) = (*solver_)(rhs_b, x_b);
        for (size_t i = 0; i < rhs.size()/2; i++)
            for (size_t j = 0; j < 2; j++)
            {
                result[2*i+j]=x_b[i](j);
            }
    }

    LinearSolverAMGCL_Block2::~LinearSolverAMGCL_Block2()
    {
    }


     LinearSolverAMGCL_Block3::LinearSolverAMGCL_Block3()
     {
         params_ = R"({
             "precond": {
                 "relax": {
                     "type": "spai0"
                 },
                 "class": "amg",
                 "direct_coarse": true,
                 "npre": 1,
                 "npost": 1,
                 "ncycle": 1,
                 "pre_cycles": 1,
                 "allow_rebuild": true,        
                 "coarse_enough": 1000,
                 "coarsening": {
                     "type": "smoothed_aggregation",
                     "estimate_spectral_radius": false,
                     "power_iters": false,
                     "relax": 1.0,
                      "aggr": {
                         "eps_strong": 0.08,
                         "block_size": 1
                     },
                      "nullspace": {
                         "cols": 0
                     }
                 }
             },
             "solver": {
                 "ns_search":false,
                 "tol": 1e-8,
                 "maxiter": 1000,
                 "type": "cg"
             }
         })"_json;

        // NOTE: usolver and psolver parameters are only used if the
        // preconditioner class is "schur_pressure_correction"
        precond_num_ = 0;
    }

    void LinearSolverAMGCL_Block3::setParameters(const json &params)
    {
        // Specially named parameters to match other solvers
        if (params.contains("max_iter"))
        {
            params_["solver"]["maxiter"] = params["max_iter"];
        }
        if (params.contains("conv_tol"))
        {
            params_["solver"]["tol"] = params["conv_tol"];
        }
        else if (params.contains("tolerance"))
        {
            params_["solver"]["tol"] = params["tolerance"];
        }
        if (params.contains("solver_type"))
        {
            params_["solver"]["type"] = params["solver_type"];
        }

        // Patch the stored params with input ones
        if (params.contains("precond"))
            params_["precond"].merge_patch(params["precond"]);
        if (params.contains("solver"))
            params_["solver"].merge_patch(params["solver"]);

        if (params_["precond"]["class"] == "schur_pressure_correction")
        {
            // Initialize the u and p solvers with a tolerance that is comparable to the main solver's
            if (!params_["precond"].contains("usolver"))
            {
                params_["precond"]["usolver"] = R"({"solver": {"maxiter": 100}})"_json;
                params_["precond"]["usolver"]["solver"]["tol"] = 10 * params_["solver"]["tol"].get<double>();
            }
            if (!params_["precond"].contains("usolver"))
            {
                params_["precond"]["psolver"] = R"({"solver": {"maxiter": 100}})"_json;
                params_["precond"]["psolver"]["solver"]["tol"] = 10 * params_["solver"]["tol"].get<double>();
            }
        }
    }

    void LinearSolverAMGCL_Block3::getInfo(json &params) const
    {
        params["num_iterations"] = iterations_;
        params["final_res_norm"] = residual_error_;
    }

    ////////////////////////////////////////////////////////////////////////////////
    void LinearSolverAMGCL_Block3::factorize(const StiffnessMatrix &Ain)
    {
        assert(precond_num_ > 0);

        long int numRows = Ain.rows();

        WrappedArray<StiffnessMatrix::StorageIndex> ia(Ain.outerIndexPtr(), numRows+1);
        WrappedArray<StiffnessMatrix::StorageIndex> ja(Ain.innerIndexPtr(), Ain.nonZeros());
        WrappedArray<StiffnessMatrix::Scalar> a(Ain.valuePtr(), Ain.nonZeros());
        std::vector<long int> _ia, _ja;
        std::vector<double> _a;
        for (size_t i = 0; i < numRows+1; i++)
        {
            _ia.push_back(ia[i]);
        }
        for (size_t i = 0; i < Ain.nonZeros(); i++)
        {
            _ja.push_back(ja[i]);
        }
        for (size_t i = 0; i < Ain.nonZeros(); i++)
        {
            _a.push_back(a[i]);
        }

        if (params_["precond"]["class"] == "schur_pressure_correction")
        {
            std::vector<char> pmask(numRows, 0);
            for (size_t i = precond_num_; i < numRows; ++i)
                pmask[i] = 1;
            params_["precond"]["pmask"] = pmask;
        }

        // AMGCL takes the parameters as a Boost property_tree (i.e., another JSON data structure)
        std::stringstream ss_params;
        ss_params << params_;
        boost::property_tree::ptree pt_params;
        boost::property_tree::read_json(ss_params, pt_params);
     
        auto A = std::tie(numRows, ia, ja, a);
        auto Ab = amgcl::adapter::block_matrix<dmat_type>(A);
        solver_ = std::make_unique<Solver>(Ab, pt_params);
        std::cout<<*solver_<<std::endl;

        iterations_ = 0;
        residual_error_ = 0;
    }

    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverAMGCL_Block3::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        assert(result.size() == rhs.size());
        std::vector<double> _rhs(rhs.data(), rhs.data() + rhs.size());
        std::vector<double> x(result.data(), result.data() + result.size());
        auto rhs_b=amgcl::backend::reinterpret_as_rhs<dmat_type>(_rhs);
        auto x_b=amgcl::backend::reinterpret_as_rhs<dmat_type>(x);

        assert(solver_ != nullptr);
        std::tie(iterations_, residual_error_) = (*solver_)(rhs_b, x_b);
        for (size_t i = 0; i < rhs.size()/3; i++)
            for (size_t j = 0; j < 3; j++)
            {
                result[3*i+j]=x_b[i](j);
            }
    }
  

    LinearSolverAMGCL_Block3::~LinearSolverAMGCL_Block3()
    {
    }
} // namespace polysolve

#endif
