//////////////////////////////////////////////////////////////////////////
#include "autodiff.h"
#include <polysolve/nonlinear/Solver.hpp>
#include <polysolve/nonlinear/BoxConstraintSolver.hpp>
#include <polysolve/nonlinear/Problem.hpp>
#include <polysolve/Utils.hpp>
#include <polysolve/Types.hpp>
#include <polysolve/linear/Solver.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <polysolve/JSONUtils.hpp>
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

public:
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

public:
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

public:
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


class InequalityConstraint : public Problem
{
public:
    InequalityConstraint(const double upper_bound): upper_bound_(upper_bound) {}
    double value(const TVector &x) override { return x(0) - upper_bound_; }
    void gradient(const TVector &x, TVector &gradv) override { gradv.setZero(x.size()); gradv(0) = 1; }
    void hessian(const TVector &x, THessian &hessian) override {}
private:
    const double upper_bound_;
};

void test_solvers(const std::vector<std::string> &solvers, const int iters, const bool exceptions_are_errors)
{
    std::vector<std::unique_ptr<TestProblem>> problems;
    problems.push_back(std::make_unique<QuadraticProblem>());
    if (!exceptions_are_errors)
        problems.push_back(std::make_unique<Rosenbrock>());
    problems.push_back(std::make_unique<Sphere>());
    problems.push_back(std::make_unique<Beale>());

    json solver_params, linear_solver_params;
    solver_params["line_search"] = {};
    solver_params["max_iterations"] = iters;

    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger");
    logger->set_level(spdlog::level::info);
    TestProblem::TVector g;
    for (auto &prob : problems)
    {
        for (auto solver_name : solvers)
        {
            if (solver_name == "BFGS" || solver_name == "DenseNewton")
                linear_solver_params["solver"] = "Eigen::LDLT";
            else
                linear_solver_params["solver"] = "Eigen::SimplicialLDLT";

            solver_params["solver"] = solver_name;

            for (const auto &ls : line_search::LineSearch::available_methods())
            {
                if (exceptions_are_errors && ls == "None")
                    continue;
                solver_params["line_search"]["method"] = ls;

                TestProblem::TVector x(prob->size());
                x.setZero();

                if (exceptions_are_errors)
                {
                    x.setRandom();
                    x /= 10;
                    x += prob->solutions()[0];
                }

                for (int i = 0; i < N_RANDOM; ++i)
                {
                    auto solver = Solver::create(solver_params,
                                                 linear_solver_params,
                                                 characteristic_length,
                                                 *logger);
                    REQUIRE(solver->name() == solver_name);

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
                        if (exceptions_are_errors)
                        {
                            INFO("solver: " + solver_name + " LS: " + ls + " problem " + prob->name());
                            CHECK(false);
                        }
                        else
                            break;
                    }

                    x.setRandom();
                    if (exceptions_are_errors)
                    {
                        x.setRandom();
                        x /= 10;
                        x += prob->solutions()[0];
                    }
                    else
                    {
                        x += prob->min();
                        x.array() *= (prob->max() - prob->min()).array();
                    }
                }
            }
        }
    }
}

TEST_CASE("non-linear", "[solver]")
{
    test_solvers(Solver::available_solvers(), 1000, false);
    // test_solvers({"L-BFGS"}, 1000, false);
}

TEST_CASE("non-linear-easier", "[solver]")
{
    test_solvers(Solver::available_solvers(), 5000, true);
}

TEST_CASE("non-linear-box-constraint", "[solver]")
{
    std::vector<std::unique_ptr<TestProblem>> problems;
    problems.push_back(std::make_unique<QuadraticProblem>());
    problems.push_back(std::make_unique<Rosenbrock>());
    problems.push_back(std::make_unique<Sphere>());
    problems.push_back(std::make_unique<Beale>());

    json solver_params, linear_solver_params;
    solver_params["box_constraints"] = {};
    solver_params["iterations_per_strategy"] = {5, 5};
    solver_params["box_constraints"]["bounds"] = std::vector<double>({{0, 4}});
    solver_params["box_constraints"]["max_change"] = 4;

    solver_params["max_iterations"] = 1000;
    solver_params["line_search"] = {};

    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger");
    logger->set_level(spdlog::level::info);

    for (auto &prob : problems)
    {
        for (auto solver_name : BoxConstraintSolver::available_solvers())
        {
            solver_params["solver"] = solver_name;

            for (const auto &ls : line_search::LineSearch::available_methods())
            {
                if (ls == "None" && solver_name != "MMA")
                    continue;
                if (solver_name == "MMA" && ls != "None")
                     continue;
                solver_params["line_search"]["method"] = ls;

                auto solver = BoxConstraintSolver::create(solver_params,
                                                          linear_solver_params,
                                                          characteristic_length,
                                                          *logger);

                REQUIRE(solver->name() == solver_name);
                QuadraticProblem::TVector x(prob->size());
                x.setConstant(3);

                for (int i = 0; i < N_RANDOM; ++i)
                {
                    try
                    {
                        solver->minimize(*prob, x);

                        INFO("solver: " + solver_params["solver"].get<std::string>() + " LS: " + ls);

                        Eigen::VectorXd gradv;
                        prob->gradient(x, gradv);
                        CHECK(solver->compute_grad_norm(x, gradv) < 1e-7);
                    }
                    catch (const std::exception &)
                    {
                        // INFO("solver: " + solver_name + " LS: " + ls + " problem " + prob->name());
                        // CHECK(false);
                        break;
                    }

                    x.setRandom();
                    x.array() += 3;
                }
            }
        }
    }
}

TEST_CASE("non-linear-box-constraint-input", "[solver]")
{
    std::vector<std::unique_ptr<TestProblem>> problems;
    problems.push_back(std::make_unique<QuadraticProblem>());

    json solver_params, linear_solver_params;
    solver_params["box_constraints"] = {};

    solver_params["max_iterations"] = 1000;
    solver_params["line_search"] = {};

    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger2");
    logger->set_level(spdlog::level::err);

    for (auto &prob : problems)
    {
        for (auto solver_name : BoxConstraintSolver::available_solvers())
        {
            solver_params["solver"] = solver_name;

            for (const auto &ls : line_search::LineSearch::available_methods())
            {
                if (ls == "None" && solver_name != "MMA")
                    continue;
                if (solver_name == "MMA" && ls != "None")
                     continue;
                solver_params["line_search"]["method"] = ls;

                QuadraticProblem::TVector x(prob->size());
                x.setConstant(3);

                Eigen::MatrixXd bounds(2, x.size());
                bounds.row(0).array() = 0;
                bounds.row(1).array() = 4;
                solver_params["box_constraints"]["bounds"] = bounds;
                Eigen::MatrixXd max_change(1, x.size());
                max_change.array() = 4;
                solver_params["box_constraints"]["max_change"] = 4;

                auto solver = BoxConstraintSolver::create(solver_params,
                                                          linear_solver_params,
                                                          characteristic_length,
                                                          *logger);
                REQUIRE(solver->name() == solver_name);

                try
                {
                    solver->minimize(*prob, x);

                    INFO("solver: " + solver_params["solver"].get<std::string>() + " LS: " + ls);

                    Eigen::VectorXd gradv;
                    prob->gradient(x, gradv);
                    CHECK(solver->compute_grad_norm(x, gradv) < 1e-7);
                }
                catch (const std::exception &)
                {
                    // INFO("solver: " + solver_name + " LS: " + ls + " problem " + prob->name());
                    // CHECK(false);
                    break;
                }
            }
        }
    }
}

TEST_CASE("MMA", "[solver]")
{
    std::vector<std::unique_ptr<TestProblem>> problems;
    problems.push_back(std::make_unique<QuadraticProblem>());
    problems.push_back(std::make_unique<Rosenbrock>());
    problems.push_back(std::make_unique<Sphere>());
    problems.push_back(std::make_unique<Beale>());

    json solver_params, linear_solver_params;
    solver_params["box_constraints"] = {};
    solver_params["box_constraints"]["bounds"] = std::vector<double>({{0, 4}});
    solver_params["box_constraints"]["max_change"] = 4;

    solver_params["max_iterations"] = 1000;
    solver_params["line_search"] = {};

    const double characteristic_length = 1;

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger");
    logger->set_level(spdlog::level::info);

    for (auto &prob : problems)
    {
        solver_params["solver"] = "MMA";
        solver_params["line_search"]["method"] = "None";

        auto solver = BoxConstraintSolver::create(solver_params,
                                                    linear_solver_params,
                                                    characteristic_length,
                                                    *logger);

        auto c = std::make_shared<InequalityConstraint>(solver_params["box_constraints"]["bounds"][1]);
        dynamic_cast<BoxConstraintSolver&>(*solver).add_constraint(c);

        QuadraticProblem::TVector x(prob->size());
        x.setConstant(3);

        for (int i = 0; i < N_RANDOM; ++i)
        {
            try
            {
                solver->minimize(*prob, x);

                INFO("solver: " + solver_params["solver"].get<std::string>());

                Eigen::VectorXd gradv;
                prob->gradient(x, gradv);
                CHECK(solver->compute_grad_norm(x, gradv) < 1e-7);
            }
            catch (const std::exception &)
            {
                // INFO("solver: " + solver_name + " LS: " + ls + " problem " + prob->name());
                // CHECK(false);
                break;
            }

            x.setRandom();
            x.array() += 3;
        }
    }
}

TEST_CASE("sample", "[solver]")
{
    Rosenbrock rb;

    Eigen::VectorXd alphas;
    Eigen::VectorXd fs;
    Eigen::VectorXi valid;

    Eigen::VectorXd dir(rb.size());
    dir.setOnes();
    for (int i = 0; i < N_RANDOM; ++i)
    {
        rb.sample_along_direction(rb.solutions()[0], dir, 0, 1, 10, alphas, fs, valid);
        dir.setRandom();

        for (int i = 1; i < fs.size(); ++i)
            CHECK(fs[0] <= fs[i]);
    }
}
