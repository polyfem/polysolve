#include "NonsmoothNewton.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace polysolve::nonlinear
{
    namespace
    {
        using Vec = Eigen::VectorXd;
        using Mat = Eigen::MatrixXd;
        using SpMat = StiffnessMatrix;

        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
        constexpr double EPS = std::numeric_limits<double>::epsilon();

        template <typename T>
        T value_or(const json &j, const char *key, const T &def)
        {
            return j.is_object() && j.contains(key) ? j[key].get<T>() : def;
        }

        /// One segment of the path (nonsmooth_segment_begin / end).
        struct SegmentScope
        {
            Problem &p;
            bool active = false;
            explicit SegmentScope(Problem &p_) : p(p_) {}
            void begin(const Vec &a, const Vec &b)
            {
                p.nonsmooth_segment_begin(a, b);
                active = true;
            }
            void end()
            {
                if (active)
                    p.nonsmooth_segment_end();
                active = false;
            }
            ~SegmentScope() { end(); }
        };
    } // namespace

    NonsmoothNewton::Options NonsmoothNewton::Options::from_json(const json &solver_params)
    {
        // parameter block NonsmoothNewton (or the parameters themselves for an entry of a strategy list)
        const json &p = solver_params.contains("NonsmoothNewton") ? solver_params["NonsmoothNewton"] : solver_params;
        Options o;
        o.max_inner = value_or(p, "max_inner", o.max_inner);
        o.relative_stop = value_or(p, "relative_stop", o.relative_stop);
        o.ftb_tau = value_or(p, "fraction_to_boundary", o.ftb_tau);
        o.rho_min = value_or(p, "rho_min", o.rho_min);
        o.max_columns = value_or(p, "max_columns", o.max_columns);
        o.handoff = value_or(p, "handoff", o.handoff);
        o.resolution = value_or(p, "resolution", o.resolution);
        if (p.is_object() && p.contains("guard"))
        {
            const json &g = p["guard"];
            o.guard = value_or(g, "enabled", o.guard);
            o.guard_c = value_or(g, "c", o.guard_c);
            o.guard_abs = value_or(g, "abs", o.guard_abs);
            o.guard_check = value_or(g, "check", o.guard_check);
        }
        if (p.is_object() && p.contains("debug"))
        {
            const json &d = p["debug"];
            o.dedupe = value_or(d, "dedupe", o.dedupe);
            o.fast_validity = value_or(d, "fast_validity", o.fast_validity);
        }
        if (o.max_inner < 1)
            throw std::runtime_error("NonsmoothNewton: max_inner must be >= 1");
        if (o.ftb_tau < 0 || o.ftb_tau >= 1)
            throw std::runtime_error("NonsmoothNewton: fraction_to_boundary must be in [0, 1)");
        if (!(o.relative_stop >= 0) || !std::isfinite(o.relative_stop))
            throw std::runtime_error("NonsmoothNewton: relative_stop must be finite and >= 0");
        // rho_min <= 0 would accept a true energy increase
        if (!(o.rho_min > 0 && o.rho_min < 1))
            throw std::runtime_error("NonsmoothNewton: rho_min must be in (0, 1), got " + std::to_string(o.rho_min));
        if (o.max_columns < 1)
            throw std::runtime_error("NonsmoothNewton: max_columns must be >= 1");
        if (o.handoff != "off" && o.handoff != "first" && o.handoff != "any")
            throw std::runtime_error("NonsmoothNewton: handoff must be off, first or any, got " + o.handoff);
        if (!(o.guard_c > 0) || !(o.guard_abs > 0) || !(o.guard_check >= 0)
            || !std::isfinite(o.guard_c) || !std::isfinite(o.guard_abs) || !std::isfinite(o.guard_check))
            throw std::runtime_error("NonsmoothNewton: guard/c and guard/abs must be finite and > 0, guard/check finite and >= 0");
        return o;
    }

    // =================================================================================================================
    struct NonsmoothNewton::Impl
    {
        struct Stats
        {
            long active_iters = 0, path_iters = 0, ls_iters = 0, path_fail = 0, failed = 0;
            long inner = 0, factorizations = 0, factorization_failures = 0, solves = 0, true_energy = 0, retractions = 0;
            long projected = 0, rebuilds = 0;
            long handoff_noise = 0, handoff_grad = 0; ///< handoffs after a noise-floor / resolution stop, on a small gradient
            long handoff_iters = 0;                   ///< direction requests declined after the handoff (each goes to the Newton chain)
        };

        /// outcome of the last take_step that went to the line search (undone if the line search fails)
        struct Pending
        {
            bool set = false;
            bool active = false;
            int inner = 0;
            long *kind = nullptr;
        };

        /// nonsmooth term at a trial point y = x + p
        struct NonsmoothEval
        {
            Vec y;        ///< trial point
            double E = 0; ///< value
            bool deriv = false;
            Vec g;                            ///< gradient
            SpMat K;                          ///< unprojected Hessian (global path)
            std::vector<HessianBlock> blocks; ///< unprojected Hessian blocks (local path)
            bool local = false;
        };

        /// reduced-space model: M-orthonormal columns Q, HQ = H_s Q, G = Q^T H_s Q, ell = Q^T g_s
        struct Model
        {
            int n = 0, k = 0, cap = 0;
            Mat Q, HQ, G;
            Vec ell;
            void init(int n_, int cap_)
            {
                n = n_;
                cap = cap_;
                k = 0;
                Q.resize(n, cap);
                HQ.resize(n, cap);
                G = Mat::Zero(cap, cap);
                ell = Vec::Zero(cap);
            }
        };

        struct ModelEval
        {
            double m = 0;
            double sumabs = 0;
            Vec grad;
            Mat A;
        };

        const Options &opts;
        spdlog::logger &logger;
        const NormType norm_type;
        double res_tol = 1e-5;
        double use_grad_norm_tol = -1; ///< line_search/use_grad_norm_tol of the solver (handoff on a small gradient)
        std::unique_ptr<polysolve::linear::Solver> linear;
        std::vector<SpMat::StorageIndex> pattern_outer, pattern_inner; ///< pattern of the last successful factorization
        Eigen::Index pattern_rows = -1;
        bool pattern_ok = false;
        Stats st;
        Pending pending;

        // per minimize
        bool fresh = true;
        Vec Mred;
        double h_edge = NaN;
        bool handed_off = false;  ///< the rest of this solve is left to the Newton chain
        bool handing_off = false; ///< the last direction request was declined by the handoff (not a failure)

        // per direction
        bool use_B = false;
        SpMat Hnc, Hcp, B;
        Vec gnc, gc0;
        double Ec0 = 0;
        bool projected = false;
        Vec last_built;          ///< point of the last nonsmooth_update done by eval (empty: unknown)
        bool noise_stop = false; ///< the last path stopped at the noise floor or below the resolution of the true energy

        Impl(const Options &o, const json &solver_params, const json &linear_solver_params, spdlog::logger &l, const NormType nt)
            : opts(o), logger(l), norm_type(nt)
        {
            res_tol = extract_param("Newton", "residual_tolerance", solver_params);
            if (solver_params.contains("line_search") && solver_params["line_search"].contains("use_grad_norm_tol"))
                use_grad_norm_tol = solver_params["line_search"]["use_grad_norm_tol"];
            linear = polysolve::linear::Solver::create(linear_solver_params, logger);
        }

        void init_minimize(Problem &p)
        {
            if (!fresh)
                return;
            // checked on first use, so that a problem without the split can still select this solver
            // the projected-mode switch relies on the factorization reporting an indefinite B (an LDLT backend would not)
            if (linear->name() != "Eigen::CholmodSupernodalLLT" && linear->name() != "Eigen::SimplicialLLT")
                log_and_throw_error(logger, "NonsmoothNewton: needs an LLT linear solver that reports indefinite matrices (Eigen::CholmodSupernodalLLT or Eigen::SimplicialLLT), got {}", linear->name());
            fresh = false;
            p.metric_diagonal(Mred);
            h_edge = p.nonsmooth_length_scale();
        }

        /// Symbolic analysis only when the sparsity pattern differs from the last successful factorization (the
        /// numeric factorization of an unchanged pattern does not depend on redoing it).
        void factorize(const SpMat &H)
        {
            if (!same_pattern(H))
            {
                pattern_ok = false;
                linear->analyze_pattern(H, H.rows());
            }
            ++st.factorizations;
            try
            {
                linear->factorize(H);
            }
            catch (...)
            {
                ++st.factorization_failures;
                pattern_ok = false;
                throw;
            }
            if (!pattern_ok && H.isCompressed())
            {
                pattern_outer.assign(H.outerIndexPtr(), H.outerIndexPtr() + H.outerSize() + 1);
                pattern_inner.assign(H.innerIndexPtr(), H.innerIndexPtr() + H.nonZeros());
                pattern_rows = H.rows();
                pattern_ok = true;
            }
        }

        bool same_pattern(const SpMat &H) const
        {
            return pattern_ok && H.isCompressed() && H.rows() == pattern_rows && H.cols() == pattern_rows
                   && size_t(H.nonZeros()) == pattern_inner.size()
                   && std::equal(pattern_outer.begin(), pattern_outer.end(), H.outerIndexPtr())
                   && std::equal(pattern_inner.begin(), pattern_inner.end(), H.innerIndexPtr());
        }

        Vec backsolve(const Vec &rhs)
        {
            ++st.solves;
            Vec out = Vec::Zero(rhs.size());
            linear->solve(rhs, out);
            return out;
        }

        double mnorm2(const Vec &v) const
        {
            return std::sqrt((Mred.array() * v.array() * v.array()).sum());
        }

        void record(long &kind, bool active, int inner)
        {
            ++kind;
            if (active)
                ++st.active_iters;
            st.inner += inner;
            pending = Pending{true, active, inner, &kind};
        }

        void undo_pending()
        {
            if (!pending.set)
                return;
            --*pending.kind;
            if (pending.active)
                --st.active_iters;
            st.inner -= pending.inner;
            pending = Pending();
        }

        // ------------------------------------------------------------------------------------------------------ direction
        /// B = H_s + proj(H_n) and the per-direction pieces (Hnc, Hcp, gnc, gc0, Ec0, B) at x.
        void assemble_parts(Problem &p, const Vec &x)
        {
            p.smooth_hessian(x, false, Hnc);
            p.smooth_gradient(x, gnc);
            p.nonsmooth_hessian(x, true, Hcp);
            B = Hnc + Hcp;
            B.makeCompressed();
            p.nonsmooth_gradient(x, gc0);
            Ec0 = p.nonsmooth_value(x);
        }

        bool try_solve(Problem &p, const SpMat &H, const Vec &g, Vec &d, const NormType nt)
        {
            try
            {
                factorize(H);
            }
            catch (const std::runtime_error &)
            {
                return false;
            }
            d = backsolve(-g);
            const double res = p.grad_norm(H * d + g, nt);
            return !(std::isnan(res) || res > res_tol);
        }

        /// B = H_s + proj(H_n) first; if its factorization fails, or p_N would need strong validity shortening, use
        /// B = proj(H_s) + proj(H_n) for this outer iteration and the model's quadratic part proj(H_s).
        bool direction_projected_mode(Problem &p, const Vec &x, const Vec &g, Vec &d)
        {
            projected = false;
            bool ok = try_solve(p, B, g, d, NormType::EUCLIDEAN);
            if (ok && opts.guard)
            {
                const double len = p.nonsmooth_step_length(x, x + d);
                if (len > opts.guard_check * h_edge && p.smooth_valid_fraction(x, x, x + d) < 0.25)
                    ok = false;
            }
            if (ok)
                return true;
            {
                SpMat h;
                p.smooth_hessian(x, true, h);
                B = h + Hcp;
                B.makeCompressed();
                Hnc = std::move(h); // projected smooth quadratic in the model (bounded below)
            }
            projected = true;
            ++st.projected;
            return try_solve(p, B, g, d, NormType::EUCLIDEAN);
        }

        // ---------------------------------------------------------------------------------------------------------- model
        void add_column(Model &M, const Vec &q)
        {
            const int j = M.k;
            M.Q.col(j) = q;
            M.HQ.col(j) = Hnc * q;
            for (int i = 0; i <= j; ++i)
            {
                const double v = M.Q.col(i).dot(M.HQ.col(j));
                M.G(i, j) = v;
                M.G(j, i) = v;
            }
            M.ell[j] = q.dot(gnc);
            ++M.k;
        }

        /// Nonsmooth term at reduced y; rebuilds its state unless reuse matches y.
        NonsmoothEval eval(Problem &p, const Vec &y, bool deriv, NonsmoothEval *reuse = nullptr)
        {
            NonsmoothEval c;
            c.y = y;
            if (reuse && reuse->y.size() == c.y.size() && reuse->y == c.y)
            {
                c.E = reuse->E;
                if (reuse->deriv)
                {
                    c.deriv = true;
                    c.g = reuse->g;
                    c.K = reuse->K;
                    c.blocks = reuse->blocks;
                    c.local = reuse->local;
                    return c;
                }
            }
            else
            {
                p.nonsmooth_update(c.y);
                last_built = c.y;
                ++st.rebuilds;
                c.E = p.nonsmooth_value(c.y);
            }
            if (deriv)
            {
                c.local = p.nonsmooth_hessian_blocks(c.y, c.g, c.blocks);
                if (!c.local)
                {
                    p.nonsmooth_gradient(c.y, c.g);
                    p.nonsmooth_hessian(c.y, false, c.K);
                }
                c.deriv = true;
            }
            return c;
        }

        ModelEval model(const Model &M, const Vec &z, const NonsmoothEval &c, bool deriv)
        {
            ModelEval e;
            const int k = M.k;
            const Vec zk = z.head(k);
            const Mat Gk = Mat(M.G.topLeftCorner(k, k));
            const Vec Gz = Gk * zk;
            const double lin = M.ell.head(k).dot(zk), quad = 0.5 * zk.dot(Gz), cc = c.E - Ec0;
            e.m = lin + quad + cc;
            e.sumabs = std::abs(lin) + std::abs(quad) + (std::abs(c.E) + std::abs(Ec0));
            if (deriv)
            {
                const Vec Hp = M.HQ.leftCols(k) * zk;
                const Vec full = gnc + Hp + c.g;
                e.grad = M.Q.leftCols(k).transpose() * full;
                if (c.local)
                {
                    e.A = Gk;
                    Eigen::Matrix<double, HessianBlock::max_size, Eigen::Dynamic> LQ(HessianBlock::max_size, k);
                    for (size_t ib = 0; ib < c.blocks.size(); ++ib)
                    {
                        const auto &b = c.blocks[ib];
                        LQ.setZero();
                        bool any = false;
                        for (int i = 0; i < b.size; ++i)
                            if (b.indices[i] >= 0)
                            {
                                LQ.row(i) = M.Q.block(b.indices[i], 0, 1, k);
                                any = true;
                            }
                        if (any)
                            e.A.noalias() += LQ.topRows(b.size).transpose() * (b.H.topLeftCorner(b.size, b.size) * LQ.topRows(b.size));
                    }
                }
                else
                {
                    const Mat KQ = c.K * M.Q.leftCols(k);
                    e.A = Gk + M.Q.leftCols(k).transpose() * KQ;
                }
                const Mat At = e.A.transpose();
                e.A = 0.5 * (e.A + At);
            }
            return e;
        }

        /// Levenberg-Marquardt step on the reduced model: sigma = 0 first, then sigma0 10^(t-1)
        bool lm_step(const Mat &A, const Vec &grad, Vec &dz)
        {
            const double anorm = A.cwiseAbs().rowwise().sum().maxCoeff();
            const double sigma0 = 1e-5 * std::max(1.0, anorm);
            Mat As = A;
            for (int t = 0; t < 18; ++t)
            {
                const double sigma = t == 0 ? 0.0 : sigma0 * std::pow(10.0, t - 1);
                As = A;
                As.diagonal().array() += sigma;
                Eigen::LLT<Mat> llt(As);
                if (llt.info() != Eigen::Success)
                    continue;
                dz = -llt.solve(grad);
                if (dz.allFinite() && grad.dot(dz) < 0)
                    return true;
            }
            return false;
        }

        double true_energy(Problem &p, const Vec &y)
        {
            ++st.true_energy;
            if (opts.dedupe)
            {
                // nonsmooth state already rebuilt at y (inside the segment): the objective there without another rebuild
                if (last_built.size() == y.size() && last_built == y)
                    return p.value(y);
            }
            p.solution_changed(y);
            last_built.resize(0);
            return p.value(y);
        }

        // ------------------------------------------------------------------------------------------------ protected path
        bool method_path(Problem &p, const Vec &x, const Vec &g, double E0, const Vec &pN, Vec &step, int &n_inner)
        {
            last_built.resize(0); // the nonsmooth state is only tracked inside this function
            noise_stop = false;
            const int inner_max = opts.max_inner;
            const int n = int(x.size());
            const int cap = std::min(opts.max_columns, inner_max) + 2;
            // smallest change of the true energy the acceptance test can resolve (0: model rounding only)
            const double e_res = opts.resolution ? 32 * EPS * std::abs(E0) : 0.0;
            const int col_cap = opts.max_columns;
            Model M;
            M.init(n, cap);

            const double npn = mnorm2(pN);
            double seg_cap = INFINITY;
            if (opts.guard)
            {
                const double pN_len = p.nonsmooth_step_length(x, x + pN);
                double alpha_v = 1;
                if (pN_len > opts.guard_check * h_edge)
                    alpha_v = p.smooth_valid_fraction(x, x, x + pN);
                seg_cap = std::min(opts.guard_c * alpha_v * pN_len, opts.guard_abs * h_edge) + p.nonsmooth_activation_distance();
            }
            if (!(npn > 0) || !std::isfinite(npn))
                return false;
            add_column(M, pN / npn);

            Vec z = Vec::Zero(cap);
            // z = 0 evaluation: the nonsmooth state is at x already
            NonsmoothEval cstate;
            cstate.y = x;
            cstate.E = Ec0;
            cstate = eval(p, x, true, &cstate); // fills derivatives only (reuse of E)

            std::vector<Vec> pts;      // p^(j) reduced displacements
            std::vector<double> mvals; // m at p^(j)
            std::vector<Vec> zs;
            pts.push_back(Vec::Zero(n));
            mvals.push_back(0.0);
            zs.push_back(Vec::Zero(cap));
            ModelEval me = model(M, z, cstate, true);
            double m_cur = 0;
            double pred_total = 0;
            Vec last_r;
            const double ell_inf = M.ell.head(M.k).cwiseAbs().maxCoeff();
            Vec dist, rate;

            for (int it = 0; it < inner_max; ++it)
            {
                if (me.grad.cwiseAbs().maxCoeff() <= 1e-10 * std::max(1.0, ell_inf))
                    break; // small reduced gradient
                Vec dz;
                if (!lm_step(me.A, me.grad, dz))
                    break; // LM failure
                int k = M.k;
                Vec zk = z.head(k);
                const Vec p_cur = M.Q.leftCols(k) * zk;
                if (opts.guard)
                {
                    const Vec pe = M.Q.leftCols(k) * (zk + dz);
                    double len = p.nonsmooth_step_length(x + p_cur, x + pe);
                    if (len > seg_cap)
                    {
                        dz *= seg_cap / len; // scale, never reject
                        len = seg_cap;
                    }
                    // with fast_validity every segment end is checked for validity; otherwise only
                    // segments longer than guard_check * h
                    if (len > opts.guard_check * h_edge || opts.fast_validity)
                    {
                        const double a = p.smooth_valid_fraction(x, x + p_cur, x + M.Q.leftCols(k) * (zk + dz));
                        if (a < 1)
                            dz *= a;
                    }
                    if (!(dz.cwiseAbs().maxCoeff() > 0))
                        break; // guard reduced the step to zero
                }
                if (opts.ftb_tau > 0)
                {
                    // fraction-to-boundary on the terms active at the segment start (linearized distance)
                    p.nonsmooth_active_distances(x + p_cur, x + M.Q.leftCols(k) * (zk + dz), dist, rate);
                    double sfac = 1;
                    for (int i = 0; i < dist.size(); ++i)
                    {
                        if (rate[i] < 0)
                        {
                            const double sc = (1 - opts.ftb_tau) * dist[i] / -rate[i];
                            sfac = std::min(sfac, sc);
                        }
                    }
                    if (sfac < 1)
                        dz *= sfac;
                }
                const Vec p_end = M.Q.leftCols(k) * (zk + dz);
                double beta;
                SegmentScope seg(p);
                // one candidate set for the whole segment
                seg.begin(x + p_cur, Vec(x + p_end));
                beta = p.max_step_size(x + p_cur, x + p_end);
                if (!(beta > 0))
                    break; // CCD step zero
                double alpha = beta >= 1 ? 1.0 : 0.99 * beta;
                const double slope = me.grad.dot(dz);
                bool found = false;
                NonsmoothEval ctrial;
                ModelEval mtrial;
                Vec ztrial;
                for (int bt = 0; bt < 50; ++bt, alpha *= 0.5)
                {
                    ztrial = z;
                    ztrial.head(k) = zk + alpha * dz;
                    const Vec y = x + M.Q.leftCols(k) * ztrial.head(k);
                    ctrial = eval(p, y, false);
                    if (!std::isfinite(ctrial.E))
                        continue;
                    mtrial = model(M, ztrial, ctrial, false);
                    if (mtrial.m <= m_cur + 1e-4 * alpha * slope + 32 * EPS * (mtrial.sumabs + me.sumabs))
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    break; // Armijo failure
                const double dec = m_cur - mtrial.m;
                if (dec <= 32 * EPS * (mtrial.sumabs + me.sumabs) + e_res)
                {
                    noise_stop = true;
                    break; // decrease below the noise floor (step not accepted)
                }
                z = ztrial;
                m_cur = mtrial.m;
                const Vec p_new = M.Q.leftCols(k) * z.head(k);
                pts.push_back(p_new);
                mvals.push_back(m_cur);
                zs.push_back(z);
                ++n_inner;
                const double prev_total = pred_total;
                pred_total = -m_cur;

                // derivatives at the accepted point (reuses the state of the last trial)
                cstate = eval(p, ctrial.y, true, &ctrial);
                seg.end();

                // ---- new column from the model residual r = grad m(p) - g - B p ----
                if (M.k < col_cap)
                {
                    const Vec r = projected ? Vec(Hnc * p_new - B * p_new + cstate.g - gc0) : Vec(cstate.g - gc0 - Hcp * p_new);
                    if (r.allFinite() && r.cwiseAbs().maxCoeff() > 0 && !(last_r.size() == r.size() && r == last_r))
                    {
                        last_r = r;
                        // M-orthogonalize u = B^-1 r against Q (two passes) and add it unless nearly dependent
                        const Vec u = backsolve(r);
                        if (u.allFinite())
                        {
                            const double nu = mnorm2(u);
                            Vec v = u;
                            for (int pass = 0; pass < 2; ++pass)
                            {
                                const Vec w = M.Q.leftCols(M.k).transpose() * (Mred.asDiagonal() * v);
                                v -= M.Q.leftCols(M.k) * w;
                            }
                            const double nv = mnorm2(v);
                            if (nu > 0 && nv / nu >= 1e-4 && M.k < cap)
                                add_column(M, v / nv);
                        }
                    }
                }

                me = model(M, z, cstate, true);
                if (opts.relative_stop > 0 && prev_total > 0 && (pred_total - prev_total) < opts.relative_stop * pred_total)
                    break; // relative model decrease below eta
            }
            if (pts.size() < 2)
                return false;

            // ---- acceptance with polyline retraction ----
            auto try_point = [&](const Vec &pt, double mval) {
                const double pred = -mval;
                if (!(pred > 0))
                    return false;
                const double Ey = true_energy(p, x + pt);
                return std::isfinite(Ey) && (E0 - Ey) / pred >= opts.rho_min;
            };
            const int K = int(pts.size()) - 1;
            // the model decreases along the polyline, so no retraction point could be verified either
            if (opts.resolution && !(-mvals[K] > e_res))
            {
                noise_stop = true;
                return false;
            }
            for (int sg = K; sg >= 1; --sg)
            {
                if (sg == K)
                {
                    if (try_point(pts[K], mvals[K]))
                    {
                        step = pts[K];
                        return true;
                    }
                }
                const Vec &pa = pts[sg - 1];
                const Vec &pb = pts[sg];
                if (sg > 1)
                {
                    for (double gam : {0.5, 0.25, 0.125})
                    {
                        ++st.retractions;
                        const Vec pt = pa + gam * (pb - pa);
                        Vec zt = zs[sg - 1] + gam * (zs[sg] - zs[sg - 1]);
                        const NonsmoothEval c = eval(p, x + pt, false);
                        const ModelEval mm = model(M, zt, c, false);
                        if (try_point(pt, mm.m))
                        {
                            step = pt;
                            return true;
                        }
                    }
                }
                else
                {
                    double gam = 0.5;
                    for (int h = 0; h < 60 && gam > 1.4e-14; ++h, gam *= 0.5)
                    {
                        ++st.retractions;
                        const Vec pt = gam * pb;
                        Vec zt = gam * zs[1];
                        const NonsmoothEval c = eval(p, x + pt, false);
                        const ModelEval mm = model(M, zt, c, false);
                        if (try_point(pt, mm.m))
                        {
                            step = pt;
                            return true;
                        }
                    }
                }
            }
            return false;
        }
    };

    // =================================================================================================================
    NonsmoothNewton::NonsmoothNewton(const json &solver_params,
                                     const json &linear_solver_params,
                                     const double characteristic_length,
                                     spdlog::logger &logger,
                                     const NormType norm_type)
        : Superclass(solver_params, characteristic_length, logger),
          opts_(Options::from_json(solver_params))
    {
        impl_ = std::make_unique<Impl>(opts_, solver_params, linear_solver_params, logger, norm_type);
    }

    NonsmoothNewton::~NonsmoothNewton() = default;

    bool NonsmoothNewton::compute_update_direction(Problem &objFunc, const TVector &x, const TVector &grad, TVector &direction)
    {
        Impl &m = *impl_;
        m.pending = Impl::Pending();
        m.use_B = false;
        // no split (e.g. a full-size phase): leave the iteration to the Newton chain that follows
        if (!objFunc.has_nonsmooth_term())
            return false;
        // handoff on a small gradient, where the line search compares gradient norms (checked before any factorization)
        if (opts_.handoff != "off" && !m.handed_off
            && objFunc.grad_norm(grad, m.norm_type) < m.use_grad_norm_tol * objFunc.grad_norm_rescaling(m.norm_type))
        {
            m.handed_off = true;
            ++m.st.handoff_grad;
        }
        if (m.handed_off)
        {
            ++m.st.handoff_iters;
            m.handing_off = true;
            return false;
        }
        m.init_minimize(objFunc);
        bool ok = false;
        try
        {
            m.assemble_parts(objFunc, x);
            ok = m.direction_projected_mode(objFunc, x, grad, direction);
        }
        catch (const std::runtime_error &e)
        {
            m_logger.debug("[{}] direction failed: {}", name(), e.what());
            ok = false;
        }
        if (!ok)
        {
            objFunc.solution_changed(x);
            return false;
        }
        m.use_B = true;
        return true;
    }

    bool NonsmoothNewton::take_step(Problem &objFunc, const TVector &x, const double energy, const TVector &grad,
                                    const TVector &direction, TVector &step)
    {
        Impl &m = *impl_;
        if (!m.use_B)
        {
            m.record(m.st.ls_iters, false, 0);
            return false;
        }
        const bool active = objFunc.nonsmooth_active_size() > 0;
        int n_inner = 0;
        m.noise_stop = false;
        const bool took_path = active && grad.dot(direction) < 0 && m.method_path(objFunc, x, grad, energy, direction, step, n_inner);
        // the true energy no longer resolves the path: the Newton chain finishes the solve from the next iteration
        if (opts_.handoff != "off" && !m.handed_off && m.noise_stop && (opts_.handoff == "any" || n_inner == 0))
        {
            m.handed_off = true;
            ++m.st.handoff_noise;
        }
        if (took_path)
        {
            m.record(m.st.path_iters, active, n_inner);
            m.pending = Impl::Pending();
            return true;
        }
        // restore the nonsmooth state at x before the line search
        objFunc.solution_changed(x);
        m.record(active ? m.st.path_fail : m.st.ls_iters, active, n_inner);
        return false;
    }

    bool NonsmoothNewton::handle_error()
    {
        // a direction declined by the handoff is not a failure: move on to the Newton chain
        if (impl_->handing_off)
        {
            impl_->handing_off = false;
            return false;
        }
        ++impl_->st.failed;
        impl_->undo_pending(); // the line search on p_N failed: the iteration does not count
        return false;
    }

    void NonsmoothNewton::reset_times()
    {
        impl_->st = Impl::Stats();
        impl_->pending = Impl::Pending();
        impl_->fresh = true;
        impl_->handed_off = false;
        impl_->handing_off = false;
    }

    void NonsmoothNewton::update_solver_info(json &solver_info, const double per_iteration)
    {
        const auto &s = impl_->st;
        const long own = s.path_iters + s.ls_iters + s.path_fail;
        const long total = solver_info.contains("iterations") ? solver_info["iterations"].get<long>() : own;
        solver_info["NonsmoothNewton"] = {
            {"active_iterations", s.active_iters},
            {"path_iterations", s.path_iters},
            {"line_search_iterations", s.ls_iters},
            {"path_failures", s.path_fail},
            {"fallback_iterations", total - own},
            {"failed_directions", s.failed},
            {"inner_steps", s.inner},
            {"factorizations", s.factorizations},
            {"factorization_failures", s.factorization_failures},
            {"back_substitutions", s.solves},
            {"true_energy_evaluations", s.true_energy},
            {"retractions", s.retractions},
            {"projected_iterations", s.projected},
            {"nonsmooth_updates", s.rebuilds},
            {"handoff_noise_floor", s.handoff_noise},
            {"handoff_grad_norm", s.handoff_grad},
            {"handoff_iterations", s.handoff_iters}};
    }
} // namespace polysolve::nonlinear
