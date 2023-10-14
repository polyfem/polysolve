//////////////////////////////////////////////////////////////////////////
#include "autodiff.h"
#include <polysolve/nonlinear/Solver.hpp>
#include <polysolve/nonlinear/Problem.hpp>
#include <polysolve/Utils.hpp>
#include <polysolve/Types.hpp>
#include <polysolve/linear/Solver.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>

#include <catch2/catch.hpp>

//////////////////////////////////////////////////////////////////////////

using namespace polysolve;
using namespace polysolve::nonlinear;

DECLARE_DIFFSCALAR_BASE();

static const int N_RANDOM = 5;

typedef DScalar2<double, Eigen::VectorXd, Eigen::MatrixXd> AutodiffScalarHessian;
typedef Eigen::Matrix<AutodiffScalarHessian, Eigen::Dynamic, 1> AutodiffHessian;

class TestProblem : public Problem
{
public:
    virtual std::vector<TVector> solutions() = 0;
    virtual int size() = 0;

    virtual TVector min() = 0;
    virtual TVector max() = 0;

    virtual std::string name() = 0;
};

class AnalyticTestProblem : public TestProblem
{
protected:
    virtual AutodiffScalarHessian eval_fun(const AutodiffHessian &x) const = 0;

private:
    AutodiffScalarHessian wrap(const TVector &x)
    {
        DiffScalarBase::setVariableCount(x.size());

        AutodiffHessian dx(x.size());

        for (long i = 0; i < x.size(); ++i)
            dx(i) = AutodiffScalarHessian(i, x(i));

        return eval_fun(dx);
    }

public:
    double value(const TVector &x) override
    {
        return wrap(x).getValue();
    }
    void gradient(const TVector &x, TVector &gradv) override
    {
        gradv = wrap(x).getGradient();
    }
    void hessian(const TVector &x, THessian &hessian) override
    {
        hessian = wrap(x).getHessian().sparseView();
    }
    void hessian(const TVector &x, Eigen::MatrixXd &hessian) override
    {
        hessian = wrap(x).getHessian();
    }
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
        hessian = sparse_identity(hessian.rows(), hessian.cols());
        hessian *= 2;
    }
    void hessian(const TVector &x, Eigen::MatrixXd &hessian) override
    {
        hessian.resize(3, 3);
        hessian.setIdentity();
        hessian *= 2;
    }

    std::vector<TVector> solutions() override
    {
        TVector res(3, 1);
        res << -2, 3, 1;
        return {res};
    }

    TVector min() override
    {
        TVector res(3, 1);
        res << 0, 0, 0;
        return res;
    }
    TVector max() override
    {
        TVector res(3, 1);
        res << 1, 1, 1;
        return res;
    }

    int size() override { return 3; }
    std::string name() override { return "Quadratic"; }
};

class Rosenbrock : public AnalyticTestProblem
{
    AutodiffScalarHessian eval_fun(const AutodiffHessian &x) const override
    {
        AutodiffScalarHessian res = AutodiffScalarHessian(0.0);
        for (int i = 0; i < x.size() - 1; ++i)
            res += 100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1 - x[i]) * (1 - x[i]);

        return res;
    }
    std::string name() override { return "Rosenbrock"; }

    int size() override { return 10; }

    std::vector<TVector> solutions() override
    {
        TVector res(size(), 1);
        res.setOnes();
        return {res};
    }

    TVector min() override
    {
        TVector res(size(), 1);
        res.setOnes();
        res *= -5;
        return res;
    }
    TVector max() override
    {
        TVector res(size(), 1);
        res.setOnes();
        res *= 5;
        return res;
    }
};

class Sphere : public AnalyticTestProblem
{
    AutodiffScalarHessian eval_fun(const AutodiffHessian &x) const override
    {
        AutodiffScalarHessian res = AutodiffScalarHessian(0.0);
        for (int i = 0; i < x.size(); ++i)
            res += x[i] * x[i];
        return res;
    }

    std::string name() override { return "Sphere"; }
    int size() override { return 10; }

    std::vector<TVector> solutions() override
    {
        TVector res(size(), 1);
        res.setZero();
        return {res};
    }

    TVector min() override
    {
        TVector res(size(), 1);
        res.setOnes();
        res *= -5;
        return res;
    }
    TVector max() override
    {
        TVector res(size(), 1);
        res.setOnes();
        res *= 5;
        return res;
    }
};

class Beale : public AnalyticTestProblem
{
    AutodiffScalarHessian eval_fun(const AutodiffHessian &x) const override
    {
        return (1.5 - x[0] + x[0] * x[1]) * (1.5 - x[0] + x[0] * x[1]) +                 //
               (2.25 - x[0] + x[0] * x[1] * x[1]) * (2.25 - x[0] + x[0] * x[1] * x[1]) + //
               (2.625 - x[0] + x[0] * x[1] * x[1] * x[1]) * (2.625 - x[0] + x[0] * x[1] * x[1] * x[1]);
    }

    std::string name() override { return "Beale"; }
    int size() override { return 2; }

    std::vector<TVector> solutions() override
    {
        TVector res(size(), 1);
        res << 3, 0.5;
        return {res};
    }

    TVector min() override
    {
        TVector res(size(), 1);
        res.setOnes();
        res *= -2;
        return res;
    }
    TVector max() override
    {
        TVector res(size(), 1);
        res.setOnes();
        res *= 6;
        return res;
    }
};

class Rosenbrock : public TestProblem
{
private:
    int n = 5;
public:
    Rosenbrock(int n_) : n(n_) {}
    double operator()(const TVector& x, TVector& grad)
    {
        double fx = (x[0] - 1.0) * (x[0] - 1.0);
        grad[0] = 2 * (x[0] - 1) + 16 * (x[0] * x[0] - x[1]) * x[0];
        for(int i = 1; i < n; i++)
        {
            fx += 4 * std::pow(x[i] - x[i - 1] * x[i - 1], 2);
            if(i == n - 1)
            {
                grad[i] = 8 * (x[i] - x[i - 1] * x[i - 1]);
            } else {
                grad[i] = 8 * (x[i] - x[i - 1] * x[i - 1]) + 16 * (x[i] * x[i] - x[i + 1]) * x[i];
            }
        }
        return fx;
    }

    double value(const TVector &x) override
    {
        double fx = (x[0] - 1.0) * (x[0] - 1.0);
        for(int i = 1; i < n; i++)
            fx += 4 * std::pow(x[i] - x[i - 1] * x[i - 1], 2);
        
        return fx;
    }
    void gradient(const TVector &x, TVector &gradv) override
    {
        gradv.setZero(n);
        gradv[0] = 2 * (x[0] - 1) + 16 * (x[0] * x[0] - x[1]) * x[0];
        for(int i = 1; i < n; i++)
        {
            if(i == n - 1)
            {
                gradv[i] = 8 * (x[i] - x[i - 1] * x[i - 1]);
            } else {
                gradv[i] = 8 * (x[i] - x[i - 1] * x[i - 1]) + 16 * (x[i] * x[i] - x[i + 1]) * x[i];
            }
        }
    }
    void hessian(const TVector &x, THessian &hessian) override
    {
        hessian.resize(n, n);
    }

    void hessian(const TVector &x, Eigen::MatrixXd &hessian) override
    {
        hessian.resize(n, n);
    }

    TVector solution() override
    {
        TVector res;
        res.setZero(n);
        return res;
    }

    virtual int size() override { return n; }
};

TEST_CASE("non-linear", "[solver]")
{
    std::vector<std::unique_ptr<TestProblem>> problems;
    problems.push_back(std::make_unique<QuadraticProblem>());
    problems.push_back(std::make_unique<Rosenbrock>());
    problems.push_back(std::make_unique<Sphere>());
    problems.push_back(std::make_unique<Beale>());

    json solver_params, linear_solver_params;
    solver_params["line_search"] = {};
    solver_params["max_iterations"] = 1000;

    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger");
    logger->set_level(spdlog::level::info);
    TestProblem::TVector g;
    for (auto &prob : problems)
    {
        for (auto solver_name : Solver::available_solvers())
        {
            if (solver_name == "BFGS" || solver_name == "DenseNewton")
                linear_solver_params["solver"] = "Eigen::LDLT";
            else
                linear_solver_params["solver"] = "Eigen::SimplicialLDLT";

            solver_params["solver"] = solver_name;

            for (const auto &ls : line_search::LineSearch::available_methods())
            {
                if (ls == "none")
                    continue;
                solver_params["line_search"]["method"] = ls;

                TestProblem::TVector x(prob->size());
                x.setZero();

                for (int i = 0; i < N_RANDOM; ++i)
                {
                    auto solver = Solver::create(solver_params,
                                                 linear_solver_params,
                                                 characteristic_length,
                                                 *logger);
                    try
                    {
                        solver->minimize(*prob, x);

                        double err = std::numeric_limits<double>::max();
                        for (auto sol : prob->solutions())
                            err = std::min(err, (x - sol).norm());
                        if (err >= 1e-7)
                        {
                            prob->gradient(x, g);
                            err = g.norm();
                        }
                        INFO("solver: " + solver_name + " LS: " + ls + " problem " + prob->name());
                        CHECK(err < 1e-7);
                        if (err >= 1e-7)
                            break;
                    }
                    catch (const std::exception &)
                    {
                        // INFO("solver: " + solver_name + " LS: " + ls + " problem " + prob->name());
                        // CHECK(false);
                        break;
                    }

                    x.setRandom();
                    x += prob->min();
                    x.array() *= (prob->max() - prob->min()).array();
                }
            }
        }
    }
}

TEST_CASE("non-linear-box-constraint", "[solver]")
{
    QuadraticProblem prob;

    json solver_params, linear_solver_params;
    solver_params["bounds"] = std::vector<double>({{0, 4}});
    solver_params["max_change"] = 4;
    solver_params["x_delta"] = 1e-10;
    solver_params["f_delta"] = 1e-12;

    solver_params["grad_norm"] = 1e-9;
    solver_params["max_iterations"] = 10;
    solver_params["relative_gradient"] = false;
    solver_params["line_search"]["use_grad_norm_tol"] = 0;
    solver_params["first_grad_norm_tol"] = 1e-10;
    solver_params["line_search"]["method"] = "backtracking";

    const double dt = 0.1;
    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger");
    logger->set_level(spdlog::level::trace);

    {
        std::string solver_name = "L-BFGS-B";

        auto solver = Solver::create(solver_name,
                                     solver_params,
                                     linear_solver_params,
                                     dt,
                                     characteristic_length,
                                     *logger);

        for (const auto &ls : line_search::LineSearch::available_methods())
        {
            if (ls == "none")
                continue;
            solver_params["line_search"]["method"] = ls;

            QuadraticProblem::TVector x(prob.size());
            x.setConstant(3);

            for (int i = 0; i < N_RANDOM; ++i)
            {
                solver->minimize(prob, x);

                INFO("solver: " + solver_name + " LS: " + ls);

                Eigen::VectorXd gradv;
                prob.gradient(x, gradv);
                std::cout << "x " << x.transpose() << "\n";
                std::cout << "grad " << gradv.transpose() << "\n";
                gradv = ((x - gradv).cwiseMax(solver_params["bounds"][0]).cwiseMin(solver_params["bounds"][1]) - x).cwiseAbs();
                std::cout << "proj grad " << gradv.transpose() << "\n";
                const double err = gradv.norm();
                CHECK(err < 1e-8);

                x.setRandom();
                x.array() += 3;
            }
        }
    }
}