//////////////////////////////////////////////////////////////////////////
#include <polysolve/nonlinear/Solver.hpp>
#include <polysolve/nonlinear/Problem.hpp>
#include <polysolve/nonlinear/descent_strategies/NonsmoothNewton.hpp>
#include <polysolve/Utils.hpp>
#include <polysolve/Types.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>
#if __has_include(<catch2/catch_test_macros.hpp>)
#include <catch2/catch_test_macros.hpp>
#else
#include <catch2/catch.hpp>
#endif

#include <cmath>
#include <limits>
#include <vector>

//////////////////////////////////////////////////////////////////////////

using namespace polysolve;
using namespace polysolve::nonlinear;

namespace
{
    /// f(x) = 1/2 |x - a|^2 + sum_i b(x_i): a quadratic pulling every coordinate below 0 and the log term
    /// b(d) = -(d - dhat)^2 log(d / dhat) for 0 < d < dhat (0 above dhat, +inf at d <= 0). The active set of the
    /// nonsmooth term is {i : x_i < dhat} at the last update.
    class LogTermProblem : public Problem
    {
    public:
        static constexpr double dhat = 0.1;
        explicit LogTermProblem(const int n) : a_(TVector::Constant(n, -1.0)) { active_.assign(n, false); }

        static double b(const double d)
        {
            if (d >= dhat)
                return 0;
            if (d <= 0)
                return std::numeric_limits<double>::infinity();
            return -(d - dhat) * (d - dhat) * std::log(d / dhat);
        }
        static double db(const double d)
        {
            return d >= dhat ? 0 : -2 * (d - dhat) * std::log(d / dhat) - (d - dhat) * (d - dhat) / d;
        }
        static double d2b(const double d)
        {
            return d >= dhat ? 0 : -2 * std::log(d / dhat) - 4 * (d - dhat) / d + (d - dhat) * (d - dhat) / (d * d);
        }

        double value(const TVector &x) override { return 0.5 * (x - a_).squaredNorm() + nonsmooth_value(x); }
        void gradient(const TVector &x, TVector &g) override
        {
            nonsmooth_gradient(x, g);
            g += x - a_;
        }
        void hessian(const TVector &x, THessian &h) override
        {
            nonsmooth_hessian(x, false, h);
            h += sparse_identity(x.size(), x.size());
        }
        void solution_changed(const TVector &x) override { nonsmooth_update(x); }
        double max_step_size(const TVector &x0, const TVector &x1) override
        {
            double t = 1;
            for (int i = 0; i < x0.size(); ++i)
                if (x1[i] < 0)
                    t = std::min(t, 0.9 * x0[i] / (x0[i] - x1[i]));
            return t;
        }

        bool has_nonsmooth_term() const override { return true; }
        void smooth_gradient(const TVector &x, TVector &g) override { g = x - a_; }
        void smooth_hessian(const TVector &x, const bool, THessian &h) override { h = sparse_identity(x.size(), x.size()); }
        double smooth_valid_fraction(const TVector &, const TVector &, const TVector &) override { return 1; }
        void metric_diagonal(TVector &d) override { d = TVector::Ones(a_.size()); }
        void nonsmooth_update(const TVector &y) override
        {
            ++updates;
            for (int i = 0; i < y.size(); ++i)
                active_[i] = y[i] < dhat;
        }
        double nonsmooth_value(const TVector &y) override
        {
            double v = 0;
            for (int i = 0; i < y.size(); ++i)
                v += b(y[i]);
            return v;
        }
        void nonsmooth_gradient(const TVector &y, TVector &g) override
        {
            g = TVector::Zero(y.size());
            for (int i = 0; i < y.size(); ++i)
                g[i] = db(y[i]);
        }
        void nonsmooth_hessian(const TVector &y, const bool project_to_psd, THessian &h) override
        {
            std::vector<Eigen::Triplet<double>> t;
            for (int i = 0; i < y.size(); ++i)
            {
                const double v = d2b(y[i]);
                if (v != 0 && (!project_to_psd || v > 0))
                    t.emplace_back(i, i, v);
            }
            h.resize(y.size(), y.size());
            h.setFromTriplets(t.begin(), t.end());
        }
        int nonsmooth_active_size() override
        {
            int n = 0;
            for (bool b : active_)
                n += b;
            return n;
        }
        void nonsmooth_active_distances(const TVector &a, const TVector &b, TVector &distance, TVector &rate) override
        {
            std::vector<double> d, r;
            for (int i = 0; i < a.size(); ++i)
                if (active_[i] && a[i] > 0)
                {
                    d.push_back(a[i]);
                    r.push_back(b[i] - a[i]);
                }
            distance = Eigen::Map<TVector>(d.data(), d.size());
            rate = Eigen::Map<TVector>(r.data(), r.size());
        }
        double nonsmooth_step_length(const TVector &a, const TVector &b) override { return (b - a).cwiseAbs().maxCoeff(); }
        double nonsmooth_length_scale() override { return 1; }
        double nonsmooth_activation_distance() override { return dhat; }

        long updates = 0;

    private:
        TVector a_;
        std::vector<bool> active_;
    };
} // namespace

TEST_CASE("nonsmooth-newton-options", "[solver][nonsmooth]")
{
    const auto d = NonsmoothNewton::Options::from_json(json::object());
    CHECK(d.max_inner == 10);
    CHECK(d.guard);
    CHECK(d.dedupe);
    CHECK(d.handoff == "any");
    CHECK(d.resolution);
    const auto o = NonsmoothNewton::Options::from_json(json{{"NonsmoothNewton", {{"max_inner", 20}, {"handoff", "first"}, {"resolution", false}, {"guard", {{"c", 3.0}}}, {"debug", {{"dedupe", false}}}}}});
    CHECK(o.max_inner == 20);
    CHECK(o.handoff == "first");
    CHECK(!o.resolution);
    CHECK(o.guard_c == 3.0);
    CHECK(o.guard_abs == 10.0);
    CHECK(!o.dedupe);
    auto bad = [](const json &p) { return json{{"NonsmoothNewton", p}}; };
    CHECK_THROWS(NonsmoothNewton::Options::from_json(bad({{"max_inner", 0}})));
    CHECK_THROWS(NonsmoothNewton::Options::from_json(bad({{"fraction_to_boundary", 1.0}})));
    // rho_min <= 0 would accept a true energy increase
    CHECK_THROWS(NonsmoothNewton::Options::from_json(bad({{"rho_min", 0.0}})));
    CHECK_THROWS(NonsmoothNewton::Options::from_json(bad({{"rho_min", 1.0}})));
    CHECK_THROWS(NonsmoothNewton::Options::from_json(bad({{"max_columns", 0}})));
    CHECK_THROWS(NonsmoothNewton::Options::from_json(bad({{"guard", {{"c", 0.0}}}})));
    CHECK_THROWS(NonsmoothNewton::Options::from_json(bad({{"handoff", "last"}})));
}

TEST_CASE("nonsmooth-newton-log-term", "[solver][nonsmooth]")
{
    // the protected path takes the steps into the region where the log term is active
    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("nonsmooth-newton-test-logger");
    logger->set_level(spdlog::level::warn);
    const json solver_params = R"({"solver": "NonsmoothNewton", "max_iterations": 100, "grad_norm_tol": 1e-10})"_json;
#ifdef POLYSOLVE_WITH_CHOLMOD
    const json linear_solver_params = R"({"solver": "Eigen::CholmodSupernodalLLT"})"_json;
#else
    const json linear_solver_params = R"({"solver": "Eigen::SimplicialLLT"})"_json;
#endif
    auto solver = Solver::create(solver_params, linear_solver_params, 1, *logger);

    LogTermProblem p(4);
    Problem::TVector x = Problem::TVector::LinSpaced(4, 0.5, 2.0);
    solver->minimize(p, x);

    Problem::TVector g;
    p.gradient(x, g);
    CHECK(g.norm() < 1e-8);
    CHECK(x.minCoeff() > 0);
    CHECK(x.maxCoeff() < LogTermProblem::dhat);
    const json &info = solver->info();
    REQUIRE(info.contains("NonsmoothNewton"));
    const json &s = info["NonsmoothNewton"];
    CHECK(s["path_iterations"].get<long>() > 0);
    CHECK(s["nonsmooth_updates"].get<long>() > 0);
    // handoff "any" (default): one handoff, after which the Newton chain finishes the solve
    CHECK(s["handoff_grad_norm"].get<long>() + s["handoff_noise_floor"].get<long>() == 1);
    CHECK(s["handoff_iterations"].get<long>() >= 1);
    CHECK(s["fallback_iterations"].get<long>() > 0);

    // without the handoff the protected path finishes the solve; the Newton chain behind it is not needed
    json no_handoff_params = solver_params;
    no_handoff_params["NonsmoothNewton"]["handoff"] = "off";
    auto solver1 = Solver::create(no_handoff_params, linear_solver_params, 1, *logger);
    LogTermProblem p1(4);
    Problem::TVector x1 = Problem::TVector::LinSpaced(4, 0.5, 2.0);
    solver1->minimize(p1, x1);
    p1.gradient(x1, g);
    CHECK(g.norm() < 1e-8);
    const json &info1 = solver1->info();
    const json &s1 = info1["NonsmoothNewton"];
    CHECK(s1["handoff_grad_norm"].get<long>() + s1["handoff_noise_floor"].get<long>() + s1["handoff_iterations"].get<long>() == 0);
    CHECK(s1["fallback_iterations"].get<long>() == 0);
    CHECK(s1["path_iterations"].get<long>() + s1["line_search_iterations"].get<long>() + s1["path_failures"].get<long>() == info1["iterations"].get<long>());

    // a large constant in the energy puts the last decreases below 32 eps |E|: with the resolution the method does not
    // try to verify them by the true energy; without it the retraction search spends true energy evaluations on them
    struct Offset : LogTermProblem
    {
        using LogTermProblem::LogTermProblem;
        double value(const TVector &x) override { return 1e8 + LogTermProblem::value(x); }
    };
    json res_params = no_handoff_params;
    json evals[2];
    for (const bool res : {true, false})
    {
        res_params["NonsmoothNewton"]["resolution"] = res;
        auto so = Solver::create(res_params, linear_solver_params, 1, *logger);
        Offset po(4);
        Problem::TVector xo = Problem::TVector::LinSpaced(4, 0.5, 2.0);
        so->minimize(po, xo);
        CHECK(xo.maxCoeff() < LogTermProblem::dhat);
        evals[res ? 0 : 1] = so->info()["NonsmoothNewton"];
    }
    CHECK(evals[0]["retractions"].get<long>() < evals[1]["retractions"].get<long>());
    CHECK(evals[0]["true_energy_evaluations"].get<long>() < evals[1]["true_energy_evaluations"].get<long>());

    // without a split the iterations are left to the Newton chain: same iterates as the Newton solver
    struct NoSplit : LogTermProblem
    {
        using LogTermProblem::LogTermProblem;
        bool has_nonsmooth_term() const override { return false; }
    } q(4);
    Problem::TVector y = Problem::TVector::LinSpaced(4, 0.5, 2.0);
    auto solver2 = Solver::create(solver_params, linear_solver_params, 1, *logger);
    solver2->minimize(q, y);
    q.gradient(y, g);
    CHECK(g.norm() < 1e-8);
    CHECK(solver2->info()["NonsmoothNewton"]["path_iterations"].get<long>() == 0);

    json newton_params = solver_params;
    newton_params["solver"] = "Newton";
    NoSplit q2(4);
    Problem::TVector y2 = Problem::TVector::LinSpaced(4, 0.5, 2.0);
    auto solver3 = Solver::create(newton_params, linear_solver_params, 1, *logger);
    solver3->minimize(q2, y2);
    CHECK(y2 == y);
    CHECK(solver3->info()["iterations"] == solver2->info()["iterations"]);

    // the linear solver requirement applies only when the split is used
    const json ldlt_params = R"({"solver": "Eigen::SimplicialLDLT"})"_json;
    NoSplit q3(4);
    Problem::TVector y3 = Problem::TVector::LinSpaced(4, 0.5, 2.0);
    auto solver4 = Solver::create(solver_params, ldlt_params, 1, *logger);
    CHECK_NOTHROW(solver4->minimize(q3, y3));
    LogTermProblem q4(4);
    Problem::TVector y4 = Problem::TVector::LinSpaced(4, 0.5, 2.0);
    auto solver5 = Solver::create(solver_params, ldlt_params, 1, *logger);
    CHECK_THROWS(solver5->minimize(q4, y4));

    // Eigen::SimplicialLLT also reports indefinite matrices (the backend of builds without CHOLMOD)
    LogTermProblem q5(4);
    Problem::TVector y5 = Problem::TVector::LinSpaced(4, 0.5, 2.0);
    auto solver6 = Solver::create(solver_params, R"({"solver": "Eigen::SimplicialLLT"})"_json, 1, *logger);
    solver6->minimize(q5, y5);
    q5.gradient(y5, g);
    CHECK(g.norm() < 1e-8);
    CHECK(solver6->info()["NonsmoothNewton"]["path_iterations"].get<long>() > 0);
}
