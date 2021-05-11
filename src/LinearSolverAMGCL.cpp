#ifdef POLYSOLVE_WITH_AMGCL

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolverAMGCL.hpp>

////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverAMGCL::LinearSolverAMGCL()
    {
        precond_num_ = 0;
        params_.solver.maxiter = 1000;
        params_.solver.tol = 1e-10;

#ifdef POLYSOLVE_AMGCL_DUMMY_PRECOND
        params_.precond.usolver.solver.tol = params_.solver.tol * 10;
        params_.precond.psolver.solver.tol = params_.solver.tol * 10;
        params_.precond.psolver.solver.maxiter = 1;
        params_.precond.usolver.solver.maxiter = 1;
#endif
#ifdef POLYSOLVE_AMGCL_V2
        params_.precond.approx_schur = false;
        params_.precond.usolver.solver.maxiter = 2;
        params_.precond.psolver.solver.maxiter = 8;
        params_.precond.usolver.solver.tol = 1e-1;
        params_.precond.psolver.solver.tol = 1e-1;
#endif
    }

    // Set solver parameters
    void LinearSolverAMGCL::setParameters(const json &params)
    {
        if (params.count("max_iter"))
        {
            params_.solver.maxiter = params["max_iter"];
        }
        // else if(params.count("pre_max_iter")) {
        // 	pre_max_iter_ = params["pre_max_iter"];
        // }
        if (params.count("conv_tol"))
        {
            params_.solver.tol = params["conv_tol"];
        }
        else if (params.count("tolerance"))
        {
            params_.solver.tol = params["tolerance"];
        }

#ifdef POLYSOLVE_AMGCL_DUMMY_PRECOND
        params_.precond.usolver.solver.tol = params_.solver.tol * 10;
        params_.precond.psolver.solver.tol = params_.solver.tol * 10;
        params_.precond.psolver.solver.maxiter = 1;
        params_.precond.usolver.solver.maxiter = 1;
#endif
    }

    void LinearSolverAMGCL::getInfo(json &params) const
    {
        params["num_iterations"] = iterations_;
        params["final_res_norm"] = residual_error_;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverAMGCL::factorize(const StiffnessMatrix &Ain)
    {
        delete solver_;
        assert(precond_num_ > 0);

        // mat = Ain;
        // solver_ = new Solver(mat, params_);

        int numRows = Ain.rows();

        ja_.resize(Ain.nonZeros());
        memcpy(ja_.data(), Ain.innerIndexPtr(), Ain.nonZeros() * sizeof(Ain.innerIndexPtr()[0]));

        ia_.resize(numRows + 1);
        memcpy(ia_.data(), Ain.outerIndexPtr(), (numRows + 1) * sizeof(Ain.outerIndexPtr()[0]));

        a_.resize(Ain.nonZeros());
        memcpy(a_.data(), Ain.valuePtr(), Ain.nonZeros() * sizeof(Ain.valuePtr()[0]));

#if defined(POLYSOLVE_AMGCL_DUMMY_PRECOND) || defined(POLYSOLVE_AMGCL_V2)
        params_.precond.pmask.resize(numRows, 0);
        for (size_t i = precond_num_; i < numRows; ++i)
            params_.precond.pmask[i] = 1;
#endif

        solver_ = new Solver(std::tie(numRows, ia_, ja_, a_), params_);

        iterations_ = 0;
        residual_error_ = 0;
    }

    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////

    void LinearSolverAMGCL::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        // result.setZero();
        // std::tie(iterations_, residual_error_) = solver_->operator()(rhs, result);

        assert(result.size() == rhs.size());

        std::vector<double> x(rhs.size()), _rhs(rhs.size());
        std::memcpy(_rhs.data(), rhs.data(), sizeof(rhs[0]) * rhs.size());
        std::memcpy(x.data(), result.data(), sizeof(x[0]) * result.size());

        std::tie(iterations_, residual_error_) = solver_->operator()(_rhs, x);

        std::memcpy(result.data(), x.data(), sizeof(x[0]) * x.size());
    }

    ////////////////////////////////////////////////////////////////////////////////

    LinearSolverAMGCL::~LinearSolverAMGCL()
    {
        delete solver_;
    }

} // namespace polysolve

#endif
