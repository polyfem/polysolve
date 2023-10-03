//////////////////////////////////////////////////////////////////////////
#include <polysolve/nonlinear/Solver.hpp>
#include <polysolve/nonlinear/Problem.hpp>
#include <polysolve/nonlinear/Utils.hpp>
#include <polysolve/Types.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>

#include <catch2/catch.hpp>

//////////////////////////////////////////////////////////////////////////

using namespace polysolve;
using namespace polysolve::nonlinear;

static const int N_RANDOM = 5;

class TestProblem : public Problem
{
public:
    virtual TVector solution() = 0;
    virtual int size() = 0;
};

// f=(x(0)+2)^2 + (x(1)-3)^2 +(x(2)-1)^2
// f'=2(x(0)+2), 2(x(1)-3), 2(x(2)-1)
// f'' identy
class QuadraticProblem : public TestProblem
{
public:
    double value(const TVector &x) override
    {
        return (x(0) + 2) * (x(0) + 2) + //
               (x(1) - 3) * (x(1) - 3) + //
               (x(2) - 1) * (x(2) - 1);
    }
    void gradient(const TVector &x, TVector &gradv) override
    {
        gradv.resize(3);
        gradv(0) = 2 * (x(0) + 2);
        gradv(1) = 2 * (x(1) - 3);
        gradv(2) = 2 * (x(2) - 1);
    }
    void hessian(const TVector &x, THessian &hessian) override
    {
        hessian.resize(3, 3);
        sparse_identity(hessian.rows(), hessian.cols());
    }

    TVector solution() override
    {
        TVector res(3, 1);
        res << -2, 3, 1;
        return res;
    }

    virtual int size() override { return 3; }
};

TEST_CASE("non-linear", "[solver]")
{
    QuadraticProblem prob;

    json solver_params, linear_solver_params;
    solver_params["x_delta"] = 1e-10;
    solver_params["f_delta"] = 1e-10;

    solver_params["grad_norm"] = 1e-8;
    solver_params["max_iterations"] = 100;
    solver_params["relative_gradient"] = false;
    solver_params["line_search"]["use_grad_norm_tol"] = 1e-8;
    solver_params["first_grad_norm_tol"] = 1e-10;
    solver_params["line_search"]["method"] = "backtracking";

    const double dt = 0.1;
    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger");

    for (auto solver_name : Solver::available_solvers())
    {
        auto solver = Solver::create(solver_name,
                                     solver_params,
                                     linear_solver_params,
                                     dt,
                                     characteristic_length,
                                     *logger);

        QuadraticProblem::TVector x(prob.size());
        x.setZero();

        for (int i = 0; i < N_RANDOM; ++i)
        {
            solver->minimize(prob, x);

            const double err = (x - prob.solution()).norm();
            INFO("solver: " + solver_name);
            REQUIRE(err < 1e-8);

            x.setRandom();
        }
    }
}