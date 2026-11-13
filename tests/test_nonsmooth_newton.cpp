//////////////////////////////////////////////////////////////////////////
#include <polysolve/nonlinear/Solver.hpp>
#include <polysolve/nonlinear/Problem.hpp>
#include <polysolve/nonlinear/descent_strategies/Newton.hpp>
#include <polysolve/nonlinear/descent_strategies/NonsmoothNewton.hpp>
#include <polysolve/Utils.hpp>
#include <polysolve/Types.hpp>

#include <nonlinear-solver-spec.hpp>

#include <jse/jse.h>

#include <spdlog/sinks/stdout_color_sinks.h>
#if __has_include(<catch2/catch_test_macros.hpp>)
#include <catch2/catch_test_macros.hpp>
#else
#include <catch2/catch.hpp>
#endif

#include <cmath>
#include <cstring>
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

        using Problem::hessian;

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
    CHECK(d.max_inner == 40);
    CHECK(d.guard);
    CHECK(d.dedupe);
    CHECK(d.handoff == "any");
    CHECK(d.resolution);
    CHECK(d.hand_back);
    CHECK(d.hand_back_energy == 100);
    CHECK(d.hand_back_contraction == 10);
    CHECK(d.jacobian_bound == 0.6);
    CHECK(d.jacobian_floor == 0.1);
    // Solver::create fills in the defaults of the spec: the same values
    {
        jse::JSE jse;
        const json rules = jse::embed::non_linear_spec::solver_spec::spec();
        json sp = R"({"solver": "NonsmoothNewton"})"_json;
        REQUIRE(jse.verify_json(sp, rules));
        sp = jse.inject_defaults(sp, rules);
        const auto ds = NonsmoothNewton::Options::from_json(sp);
        CHECK(ds.max_inner == d.max_inner);
        CHECK(ds.relative_stop == d.relative_stop);
        CHECK(ds.ftb_tau == d.ftb_tau);
        CHECK(ds.rho_min == d.rho_min);
        CHECK(ds.max_columns == d.max_columns);
        CHECK(ds.handoff == d.handoff);
        CHECK(ds.resolution == d.resolution);
        CHECK(ds.hand_back == d.hand_back);
        CHECK(ds.hand_back_energy == d.hand_back_energy);
        CHECK(ds.hand_back_contraction == d.hand_back_contraction);
        CHECK(ds.jacobian_bound == d.jacobian_bound);
        CHECK(ds.jacobian_floor == d.jacobian_floor);
        CHECK(ds.guard == d.guard);
        CHECK(ds.guard_c == d.guard_c);
        CHECK(ds.guard_abs == d.guard_abs);
        CHECK(ds.guard_check == d.guard_check);
        CHECK(ds.dedupe == d.dedupe);
        CHECK(ds.fast_validity == d.fast_validity);
    }
    const auto o = NonsmoothNewton::Options::from_json(json{{"NonsmoothNewton", {{"max_inner", 20}, {"handoff", "first"}, {"resolution", false}, {"guard", {{"c", 3.0}}}, {"hand_back", {{"enabled", true}, {"energy", 50.0}}}, {"jacobian", {{"bound", 0.5}}}, {"debug", {{"dedupe", false}}}}}});
    CHECK(o.max_inner == 20);
    CHECK(o.handoff == "first");
    CHECK(!o.resolution);
    CHECK(o.guard_c == 3.0);
    CHECK(o.guard_abs == 10.0);
    CHECK(o.hand_back);
    CHECK(o.hand_back_energy == 50);
    CHECK(o.hand_back_contraction == 10);
    CHECK(o.jacobian_bound == 0.5);
    CHECK(o.jacobian_floor == 0.1);
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
    CHECK_THROWS(NonsmoothNewton::Options::from_json(bad({{"hand_back", {{"energy", -1.0}}}})));
    CHECK_THROWS(NonsmoothNewton::Options::from_json(bad({{"hand_back", {{"contraction", 0.5}}}})));
    CHECK_THROWS(NonsmoothNewton::Options::from_json(bad({{"jacobian", {{"bound", 1.0}}}})));
    CHECK_THROWS(NonsmoothNewton::Options::from_json(bad({{"jacobian", {{"bound", -0.1}}}})));
    CHECK_THROWS(NonsmoothNewton::Options::from_json(bad({{"jacobian", {{"floor", -1.0}}}})));
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

TEST_CASE("nonsmooth-newton-hand-back", "[solver][nonsmooth]")
{
    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("nonsmooth-newton-hand-back-logger");
    logger->set_level(spdlog::level::warn);
#ifdef POLYSOLVE_WITH_CHOLMOD
    const json linear_solver_params = R"({"solver": "Eigen::CholmodSupernodalLLT"})"_json;
#else
    const json linear_solver_params = R"({"solver": "Eigen::SimplicialLLT"})"_json;
#endif
    // the parameters the strategy reads, as Solver::create passes them
    const json params = {{"NonsmoothNewton", {{"hand_back", {{"enabled", true}}}}},
                         {"Newton", {{"residual_tolerance", 1e-5}}},
                         {"line_search", {{"use_grad_norm_tol", 1e-4}}}};

    LogTermProblem p(4);
    const Problem::TVector x = Problem::TVector::LinSpaced(4, 0.02, 0.08); // every term active
    p.solution_changed(x);
    const Problem::TVector small_g = Problem::TVector::Constant(4, 1e-6); // below use_grad_norm_tol
    Problem::TVector d;

    const auto counter = [](NonsmoothNewton &s, const char *name) {
        json info;
        s.update_solver_info(info, 1);
        return info["NonsmoothNewton"][name].get<long>();
    };
    // a strategy that has just handed off on a small gradient
    const auto handed_off = [&](const json &prm) {
        auto s = std::make_unique<NonsmoothNewton>(prm, linear_solver_params, 1, *logger, NormType::EUCLIDEAN);
        s->reset_times();
        REQUIRE(!s->compute_update_direction(p, x, small_g, d));
        REQUIRE(!s->handle_error());
        REQUIRE(counter(*s, "handoff_grad_norm") == 1);
        return s;
    };

    // chain step: unconverged, energy decrease far above the resolution 32 eps |E|
    const double E0 = 1, E1 = 0.5, g0 = 1, tol = 1e-8;
    {
        // cut by its line search: the iterations come back, and the gradient handoff stays off for the solve
        auto s = handed_off(params);
        s->chain_step_result(1, 0.5, E0, E1, g0, g0 / 2, tol);
        CHECK(counter(*s, "hand_backs") == 1);
        CHECK(s->compute_update_direction(p, x, small_g, d));
        CHECK(counter(*s, "handoff_grad_norm") == 1);
        CHECK(counter(*s, "handoff_iterations") == 1);
        s->chain_step_result(1, 0.5, E0, E1, g0, g0 / 2, tol);
        CHECK(counter(*s, "hand_backs") == 1);
    }

    struct Step
    {
        int strategy;
        double rate, energy_after, grad_norm_after;
    };
    // each of the three signs of a slow step hands back on its own
    for (const Step &c : {Step{1, 0.5, E1, g0 / 16}, Step{2, 1.0, E1, g0 / 16}, Step{1, 1.0, E1, g0 / 4}})
    {
        auto s = handed_off(params);
        s->chain_step_result(c.strategy, c.rate, E0, c.energy_after, g0, c.grad_norm_after, tol);
        CHECK(counter(*s, "hand_backs") == 1);
    }
    // converged; energy decrease below 100 times the resolution; a full Newton step that contracts the gradient 16 times
    const double eps = std::numeric_limits<double>::epsilon();
    for (const Step &c : {Step{1, 0.5, E1, tol / 2}, Step{1, 0.5, E0 - 50 * 32 * eps * E0, g0 / 2}, Step{1, 1.0, E1, g0 / 16}})
    {
        auto s = handed_off(params);
        s->chain_step_result(c.strategy, c.rate, E0, c.energy_after, g0, c.grad_norm_after, tol);
        CHECK(counter(*s, "hand_backs") == 0);
    }

    // off, or no handoff in this solve: nothing to hand back
    {
        json off = params;
        off["NonsmoothNewton"]["hand_back"]["enabled"] = false;
        auto s = handed_off(off);
        s->chain_step_result(1, 0.5, E0, E1, g0, g0 / 2, tol);
        CHECK(counter(*s, "hand_backs") == 0);
        NonsmoothNewton fresh(params, linear_solver_params, 1, *logger, NormType::EUCLIDEAN);
        fresh.reset_times();
        fresh.chain_step_result(1, 0.5, E0, E1, g0, g0 / 2, tol);
        CHECK(counter(fresh, "hand_backs") == 0);
    }

    // after the handoff at the noise floor the true energy cannot verify the path any more: no hand-back
    {
        struct Offset : LogTermProblem
        {
            using LogTermProblem::LogTermProblem;
            double value(const TVector &x) override { return 1e8 + LogTermProblem::value(x); }
        } po(4);
        // a point next to the minimizer: the decrease of the path is far below 32 eps |E| = 7e-7
        LogTermProblem q(4);
        Problem::TVector xs = Problem::TVector::LinSpaced(4, 0.5, 2.0);
        auto newton = Solver::create(R"({"solver": "Newton", "max_iterations": 100, "grad_norm_tol": 1e-10})"_json, linear_solver_params, 1, *logger);
        newton->minimize(q, xs);
        xs.array() += 1e-6;
        po.solution_changed(xs);
        Problem::TVector gs, step;
        po.gradient(xs, gs);
        json no_grad_handoff = params;
        no_grad_handoff.erase("line_search");
        NonsmoothNewton s(no_grad_handoff, linear_solver_params, 1, *logger, NormType::EUCLIDEAN);
        s.reset_times();
        REQUIRE(s.compute_update_direction(po, xs, gs, d));
        CHECK(!s.take_step(po, xs, po.value(xs), gs, d, step));
        REQUIRE(counter(s, "handoff_noise_floor") == 1);
        s.chain_step_result(1, 0.5, E0, E1, g0, g0 / 2, tol);
        CHECK(counter(s, "hand_backs") == 0);
    }
}

TEST_CASE("nonsmooth-newton-jacobian-limits", "[solver][nonsmooth]")
{
    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("nonsmooth-newton-jacobian-logger");
    logger->set_level(spdlog::level::warn);
#ifdef POLYSOLVE_WITH_CHOLMOD
    const json linear_solver_params = R"({"solver": "Eigen::CholmodSupernodalLLT"})"_json;
#else
    const json linear_solver_params = R"({"solver": "Eigen::SimplicialLLT"})"_json;
#endif
    // the coordinates of the log-term problem as elements with volume ratio J_i(y) = y_i
    struct JacobianProblem : LogTermProblem
    {
        using LogTermProblem::LogTermProblem;
        long calls = 0;
        double smooth_min_jacobian(const TVector &x) override
        {
            ++calls;
            return x.minCoeff();
        }
        void smooth_min_jacobians(const TVector &x, const TVector &y, const bool want_min, const bool want_ratio, double &min_j, double &min_ratio) override
        {
            ++calls;
            min_j = want_min ? y.minCoeff() : std::numeric_limits<double>::quiet_NaN();
            min_ratio = want_ratio ? (y.array() / x.array()).minCoeff() : std::numeric_limits<double>::quiet_NaN();
        }
    };
    const auto params = [](const json &jacobian) {
        return json{{"NonsmoothNewton", {{"jacobian", jacobian}}}, {"Newton", {{"residual_tolerance", 1e-5}}}};
    };
    const auto counter = [](NonsmoothNewton &s, const char *name) {
        json info;
        s.update_solver_info(info, 1);
        return info["NonsmoothNewton"][name].get<long>();
    };
    // one outer iteration at x: the accepted point of the path (x itself if the path failed)
    const auto try_outer = [](Problem &p, NonsmoothNewton &s, const Problem::TVector &x, bool &took) {
        p.solution_changed(x);
        Problem::TVector g, d, step;
        p.gradient(x, g);
        REQUIRE(s.compute_update_direction(p, x, g, d));
        took = s.take_step(p, x, p.value(x), g, d, step);
        return took ? Problem::TVector(x + step) : x;
    };
    const auto outer = [&](Problem &p, NonsmoothNewton &s, const Problem::TVector &x) {
        bool took = false;
        const Problem::TVector y = try_outer(p, s, x, took);
        REQUIRE(took);
        return y;
    };

    // every term active; the minimizer is near 0.012 in every coordinate, so the paths compress strongly
    const Problem::TVector x0 = Problem::TVector::LinSpaced(4, 0.05, 0.08);
    JacobianProblem p0(4);
    NonsmoothNewton free_(params({{"bound", 0.0}, {"floor", 0.0}}), linear_solver_params, 1, *logger, NormType::EUCLIDEAN);
    free_.reset_times();
    const Problem::TVector y_free = outer(p0, free_, x0);
    CHECK(p0.calls == 0); // limits off: no callbacks
    CHECK((y_free.array() / x0.array()).minCoeff() < 0.6);
    CHECK(counter(free_, "inner_steps") > 2);

    {
        // bound: the first step would cross it, so it is halved until it holds and the path ends after it
        JacobianProblem p(4);
        NonsmoothNewton s(params({{"bound", 0.6}, {"floor", 0.0}}), linear_solver_params, 1, *logger, NormType::EUCLIDEAN);
        s.reset_times();
        const Problem::TVector y = outer(p, s, x0);
        CHECK((y.array() / x0.array()).minCoeff() >= 0.6);
        CHECK(counter(s, "jacobian_bound_hits") == 1);
        CHECK(counter(s, "jacobian_floor_hits") == 0);
        CHECK(counter(s, "inner_steps") == 1);
    }
    {
        // a lower bound lets the first step through and stops the path at a later step: it ends at the previous point
        JacobianProblem p(4);
        NonsmoothNewton s(params({{"bound", 0.15}, {"floor", 0.0}}), linear_solver_params, 1, *logger, NormType::EUCLIDEAN);
        s.reset_times();
        const Problem::TVector y = outer(p, s, x0);
        CHECK((y.array() / x0.array()).minCoeff() >= 0.15);
        CHECK(counter(s, "jacobian_bound_hits") == 1);
        CHECK(counter(s, "inner_steps") >= 1);
        CHECK(counter(s, "inner_steps") < counter(free_, "inner_steps"));
    }
    {
        // floor: J_ref = min J at the first direction of the solve (0.05), so min J stays >= 0.9 * 0.05
        JacobianProblem p(4);
        NonsmoothNewton s(params({{"bound", 0.0}, {"floor", 0.9}}), linear_solver_params, 1, *logger, NormType::EUCLIDEAN);
        s.reset_times();
        const Problem::TVector y = outer(p, s, x0);
        CHECK(y.minCoeff() >= 0.9 * x0.minCoeff());
        CHECK(y.minCoeff() < x0.minCoeff());
        CHECK(counter(s, "jacobian_floor_hits") == 1);
        CHECK(counter(s, "jacobian_bound_hits") == 0);
        // an outer iterate already below the floor: no inner point may lower min J further; here every step along
        // the path does, so the path fails and the Newton step is line searched
        const Problem::TVector x1 = Problem::TVector::LinSpaced(4, 0.03, 0.05);
        bool took = true;
        try_outer(p, s, x1, took);
        CHECK(!took);
        CHECK(counter(s, "path_failures") == 1);
        CHECK(counter(s, "jacobian_floor_hits") == 2);
        // the reference belongs to the solve: from x1 as the first iterate, the floor is 0.9 * 0.03
        s.reset_times();
        const Problem::TVector y2 = outer(p, s, x1);
        CHECK(y2.minCoeff() >= 0.9 * x1.minCoeff());
        CHECK(y2.minCoeff() < x1.minCoeff());
        CHECK(counter(s, "jacobian_floor_hits") == 1);
    }
    {
        // a problem without the callbacks (NaN): no limits, the same path as with the limits off
        LogTermProblem p(4);
        NonsmoothNewton s(params({{"bound", 0.6}, {"floor", 0.1}}), linear_solver_params, 1, *logger, NormType::EUCLIDEAN);
        s.reset_times();
        const Problem::TVector y = outer(p, s, x0);
        CHECK(y == y_free);
        CHECK(counter(s, "inner_steps") == counter(free_, "inner_steps"));
        CHECK(counter(s, "jacobian_bound_hits") + counter(s, "jacobian_floor_hits") == 0);
    }
    {
        // a whole solve with both limits converges
        const json solver_params = {{"solver", "NonsmoothNewton"}, {"max_iterations", 100}, {"grad_norm_tol", 1e-10},
                                    {"NonsmoothNewton", {{"jacobian", {{"bound", 0.6}, {"floor", 0.1}}}}}};
        auto solver = Solver::create(solver_params, linear_solver_params, 1, *logger);
        JacobianProblem p(4);
        Problem::TVector x = x0;
        solver->minimize(p, x);
        Problem::TVector g;
        p.gradient(x, g);
        CHECK(g.norm() < 1e-8);
        CHECK(solver->info()["NonsmoothNewton"]["jacobian_bound_hits"].get<long>() > 0);
    }
}

TEST_CASE("nonlinear-chain-step-result", "[solver][nonsmooth]")
{
    // a first strategy that declines every direction: the Newton chain behind it takes all steps, and the first strategy
    // sees each of them at the start of the next iteration
    struct Recorder : DescentStrategy
    {
        struct Call
        {
            int strategy;
            double rate, energy_before, energy_after, grad_norm_before, grad_norm_after, grad_tol;
        };
        std::vector<Call> calls;

        using DescentStrategy::DescentStrategy;
        std::string name() const override { return "Recorder"; }
        bool compute_update_direction(Problem &, const TVector &, const TVector &, TVector &) override { return false; }
        void chain_step_result(const int strategy, const double rate, const double energy_before, const double energy_after,
                               const double grad_norm_before, const double grad_norm_after, const double grad_tol) override
        {
            calls.push_back({strategy, rate, energy_before, energy_after, grad_norm_before, grad_norm_after, grad_tol});
        }
    };
    struct NoSplit : LogTermProblem
    {
        using LogTermProblem::LogTermProblem;
        bool has_nonsmooth_term() const override { return false; }
    };

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("chain-step-test-logger");
    logger->set_level(spdlog::level::warn);
    json params = R"({"solver": "Newton", "max_iterations": 100, "grad_norm_tol": 1e-10})"_json;
    jse::JSE jse;
    const json rules = jse::embed::non_linear_spec::solver_spec::spec();
    REQUIRE(jse.verify_json(params, rules));
    params = jse.inject_defaults(params, rules);
#ifdef POLYSOLVE_WITH_CHOLMOD
    const json linear_solver_params = R"({"solver": "Eigen::CholmodSupernodalLLT"})"_json;
#else
    const json linear_solver_params = R"({"solver": "Eigen::SimplicialLLT"})"_json;
#endif

    Solver solver(params, 1, *logger);
    auto recorder = std::make_shared<Recorder>(params, 1, *logger);
    solver.add_strategy(recorder);
    for (const auto &s : Newton::create_solver(true, params, linear_solver_params, 1, *logger, NormType::EUCLIDEAN, 1))
        solver.add_strategy(s);
    solver.set_strategies_iterations(params);

    NoSplit p(4);
    Problem::TVector x = Problem::TVector::LinSpaced(4, 0.5, 2.0);
    const double e0 = p.value(x);
    solver.minimize(p, x);

    // every step is reported, the last one at the start of the iteration that converges
    const auto &c = recorder->calls;
    REQUIRE(!c.empty());
    CHECK(long(c.size()) == solver.info()["iterations"].get<long>());
    CHECK(c.front().energy_before == e0);
    for (size_t i = 0; i < c.size(); ++i)
    {
        CHECK(c[i].strategy >= 1);
        CHECK(c[i].rate > 0);
        CHECK(c[i].rate <= 1);
        CHECK(c[i].energy_after <= c[i].energy_before);
        CHECK(c[i].grad_tol == 1e-10);
        if (i + 1 < c.size())
        {
            CHECK(c[i + 1].energy_before == c[i].energy_after);
            CHECK(c[i + 1].grad_norm_before == c[i].grad_norm_after);
        }
    }
    CHECK(c.back().grad_norm_after <= c.back().grad_tol);
    CHECK(c.back().energy_after == p.value(x));
}

TEST_CASE("newton-analysis-reuse", "[solver][newton]")
{
    // E(x) = 1/2 x^T K x - w sum_i cos(x_i) with K tridiagonal (2, -0.5): the Hessian K + w diag(cos x_i) has the same
    // pattern everywhere and is indefinite where some cos x_i is close to -1
    class WavyProblem : public Problem
    {
    public:
        explicit WavyProblem(const double w) : w_(w) {}
        using Problem::hessian;
        double value(const TVector &x) override
        {
            return 0.5 * x.dot(K(x.size()) * x) - w_ * x.array().cos().sum();
        }
        void gradient(const TVector &x, TVector &g) override
        {
            g = K(x.size()) * x;
            g.array() += w_ * x.array().sin();
        }
        void hessian(const TVector &x, THessian &h) override
        {
            h = K(x.size());
            for (int i = 0; i < x.size(); ++i)
                h.coeffRef(i, i) += w_ * std::cos(x[i]);
            h.makeCompressed();
        }

    private:
        static THessian K(const Eigen::Index n)
        {
            std::vector<Eigen::Triplet<double>> t;
            for (int i = 0; i < n; ++i)
            {
                t.emplace_back(i, i, 2.0);
                if (i + 1 < n)
                {
                    t.emplace_back(i, i + 1, -0.5);
                    t.emplace_back(i + 1, i, -0.5);
                }
            }
            THessian k(n, n);
            k.setFromTriplets(t.begin(), t.end());
            return k;
        }
        const double w_;
    };

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("newton-analysis-reuse-logger");
    logger->set_level(spdlog::level::warn);
    const json params = {{"Newton", {{"residual_tolerance", 1e-5}}}};
#ifdef POLYSOLVE_WITH_CHOLMOD
    const std::vector<std::string> solvers = {"Eigen::CholmodSupernodalLLT", "Eigen::SimplicialLLT", "Eigen::SimplicialLDLT"};
#else
    const std::vector<std::string> solvers = {"Eigen::SimplicialLLT", "Eigen::SimplicialLDLT"};
#endif

    const int n = 50;
    WavyProblem p(3.0);
    const Problem::TVector xa = Problem::TVector::Constant(n, 0.3), xb = Problem::TVector::LinSpaced(n, -0.5, 0.5);
    Problem::TVector bad = xa; // w cos(pi) = -3 makes the Hessian indefinite
    bad[n / 2] = std::acos(-1.0);
    const std::vector<const Problem::TVector *> points = {&xa, &bad, &xb, &bad, &xa, &xb};
    for (const std::string &ls : solvers)
    {
        const json linear_solver_params = {{"solver", ls}};
        // one strategy that keeps its analysis over all points (after a failed factorization too) against a new strategy
        // at every point: the same directions bit for bit
        Newton reused(true, params, linear_solver_params, 1, *logger, NormType::EUCLIDEAN);
        int n_ok = 0;
        for (const Problem::TVector *x : points)
        {
            // the strategy writes the direction into a vector of the problem's size (as Solver::minimize passes it)
            Problem::TVector g, d = Problem::TVector::Zero(n), d_new = Problem::TVector::Zero(n);
            p.gradient(*x, g);
            const bool ok = reused.compute_update_direction(p, *x, g, d);
            Newton fresh(true, params, linear_solver_params, 1, *logger, NormType::EUCLIDEAN);
            const bool ok_new = fresh.compute_update_direction(p, *x, g, d_new);
            CHECK(ok == ok_new);
            if (ls != "Eigen::SimplicialLDLT") // LDLT factors an indefinite matrix
                CHECK(ok == (x != &bad));
            if (ok && ok_new)
            {
                ++n_ok;
                REQUIRE(d.size() == d_new.size());
                CHECK(std::memcmp(d.data(), d_new.data(), sizeof(double) * d.size()) == 0);
            }
        }
        CHECK(n_ok >= 4);
    }
}
