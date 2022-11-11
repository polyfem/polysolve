#ifdef POLYSOLVE_WITH_AMGCL
#ifdef POLYSOLVE_WITH_CUDA

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolverAMGCL_cuda.hpp>

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

        json default_params()
        {
            json params = R"({
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

            return params;
        }

        void set_params(const json &params, json &out)
        {
            if (params.contains("AMGCL_cuda"))
            {
                // Patch the stored params with input ones
                if (params["AMGCL_cuda"].contains("precond"))
                    out["precond"].merge_patch(params["AMGCL_cuda"]["precond"]);
                if (params["AMGCL_cuda"].contains("solver"))
                    out["solver"].merge_patch(params["AMGCL_cuda"]["solver"]);

                if (out["precond"]["class"] == "schur_pressure_correction")
                {
                    // Initialize the u and p solvers with a tolerance that is comparable to the main solver's
                    if (!out["precond"].contains("usolver"))
                    {
                        out["precond"]["usolver"] = R"({"solver": {"maxiter": 100}})"_json;
                        out["precond"]["usolver"]["solver"]["tol"] = 10 * out["solver"]["tol"].get<double>();
                    }
                    if (!out["precond"].contains("usolver"))
                    {
                        out["precond"]["psolver"] = R"({"solver": {"maxiter": 100}})"_json;
                        out["precond"]["psolver"]["solver"]["tol"] = 10 * out["solver"]["tol"].get<double>();
                    }
                }
            }
        }
    } // namespace

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverAMGCL_cuda::LinearSolverAMGCL_cuda()
    {
        params_ = default_params();
        // NOTE: usolver and psolver parameters are only used if the
        // preconditioner class is "schur_pressure_correction"
        precond_num_ = 0;
        cusparseCreate(&backend_params_.cusparse_handle);
    }

    // Set solver parameters
    void LinearSolverAMGCL_cuda::setParameters(const json &params)
    {
        if (params.contains("AMGCL_cuda"))
        {
            set_params(params, params_);
        }
    }

    void LinearSolverAMGCL_cuda::getInfo(json &params) const
    {
        params["num_iterations"] = iterations_;
        params["final_res_norm"] = residual_error_;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverAMGCL_cuda::factorize(const StiffnessMatrix &Ain)
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
        auto A = std::tie(numRows, ia, ja, a);
        solver_ = std::make_unique<Solver>(A, pt_params, backend_params_);
        // std::cout << *solver_.get() << std::endl;
        iterations_ = 0;
        residual_error_ = 0;
    }

    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverAMGCL_cuda::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        assert(result.size() == rhs.size());
        std::vector<double> _rhs(rhs.data(), rhs.data() + rhs.size());
        std::vector<double> x(result.data(), result.data() + result.size());
        auto rhs_b = Backend::copy_vector(_rhs, backend_params_);
        auto x_b = Backend::copy_vector(x, backend_params_);

        std::tie(iterations_, residual_error_) = (*solver_)(*rhs_b, *x_b);
        thrust::copy(x_b->begin(), x_b->end(), result.data());
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverAMGCL_cuda::~LinearSolverAMGCL_cuda()
    {
    }

} // namespace polysolve

#endif
#endif
