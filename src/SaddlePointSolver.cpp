#include <polysolve/SaddlePointSolver.hpp>

#include <unsupported/Eigen/SparseExtra>

#include <ctime>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{
    namespace
    {
        void diag(const Eigen::VectorXd &V, StiffnessMatrix &X)
        {
            // clear and resize output
            Eigen::DynamicSparseMatrix<double, Eigen::RowMajor> dyn_X(V.size(), V.size());
            dyn_X.reserve(V.size());
            // loop over non-zeros
            for (int i = 0; i < V.size(); i++)
            {
                dyn_X.coeffRef(i, i) += V[i];
            }
            X = StiffnessMatrix(dyn_X);
        }

        void compute_solution(const int i, const Eigen::VectorXd &alphau, const Eigen::VectorXd &alphap,
                              const std::vector<Eigen::VectorXd> &yu, const std::vector<Eigen::VectorXd> &yp,
                              const StiffnessMatrix &Wm, const StiffnessMatrix &Wc,
                              Eigen::Ref<Eigen::VectorXd> result)
        {
            // yu = alphau(1) * iters{1}.yu;
            // yp = alphap(1) * iters{1}.yp;
            Eigen::VectorXd yuf = alphau(0) * yu[0];
            Eigen::VectorXd ypf = alphap(0) * yp[0];

            for (int j = 1; j < i; ++j)
            {
                // yu = yu + alphau(j) * iters{j}.yu;
                // yp = yp + alphap(j) * iters{j}.yp;

                yuf += alphau(j) * yu[j];
                ypf += alphap(j) * yp[j];
            }

            yuf = Wm * yuf;
            ypf = Wc * ypf;

            result.resize(yuf.size() + ypf.size());
            result.topRows(yuf.size()) = yuf;
            result.bottomRows(ypf.size()) = ypf;
        }
    } // namespace

    ////////////////////////////////////////////////////////////////////////////////

    SaddlePointSolver::SaddlePointSolver()
    {
        conv_tol_ = 1e-8;
        max_iter_ = 50;

#ifdef POLYSOLVE_WITH_PARDISO
        asymmetric_solver_name_ = "HypreGMRES";
        symmetric_solver_name_ = LinearSolver::defaultSolver();
#else
        asymmetric_solver_name_ = "HypreGMRES";
        symmetric_solver_name_ = "HypreGMRES";
#endif

        asymmetric_solver_params_ = {"tolerance", 1e-5};
        symmetric_solver_params_ = {"tolerance", 1e-5};
    }

    // Set solver parameters
    void SaddlePointSolver::setParameters(const json &params)
    {
        if (params.count("max_iter"))
        {
            max_iter_ = params["max_iter"];
        }

        if (params.count("conv_tol"))
        {
            conv_tol_ = params["conv_tol"];
        }
        else if (params.count("tolerance"))
        {
            conv_tol_ = params["tolerance"];
        }

        if (params.count("asymmetric_solver_name"))
        {
            asymmetric_solver_name_ = params["asymmetric_solver_name"].get<std::string>();
        }

        if (params.count("asymmetric_solver_params"))
        {
            asymmetric_solver_params_ = params["asymmetric_solver_params"];
        }

        if (params.count("symmetric_solver_name"))
        {
            symmetric_solver_name_ = params["symmetric_solver_name"].get<std::string>();
        }

        if (params.count("symmetric_solver_params"))
        {
            symmetric_solver_params_ = params["symmetric_solver_params"];
        }
    }

    void SaddlePointSolver::getInfo(json &params) const
    {
        params["num_iterations"] = num_iterations_;
        params["final_res_norm"] = final_res_norm_;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void SaddlePointSolver::analyzePattern(const StiffnessMatrix &Ain, const int precond_num)
    {
        Ain_ = Ain;
        // A = M.A(1:ablock-1, 1:ablock-1);
        // B = M.A(1:ablock-1, ablock:end);
        // Bt = M.A(ablock:end, 1:ablock-1);
        // C = M.A(ablock:end, ablock:end);

        precond_num_ = precond_num;

        const int other_size = Ain.rows() - precond_num;

        const StiffnessMatrix A = Ain.block(0, 0, precond_num, precond_num);
        const StiffnessMatrix B = Ain.block(0, precond_num, precond_num, other_size);
        const StiffnessMatrix C = Ain.block(precond_num, precond_num, other_size, other_size);

        // Wm = spdiags(sqrt(1./diag(A)), 0, length(A), length(A));
        // Wc = spdiags(sqrt(1./diag(C)), 0, length(C), length(C));
        Eigen::VectorXd Wmd = A.diagonal();
        Eigen::VectorXd Wcd = C.diagonal();
        Wmd = (1. / Wmd.array().sqrt()).eval();
        // Wcd = (1. / Wcd.array().sqrt()).eval();

        diag(Wmd, Wm);
        // diag(Wcd, Wc);
        Wc.resize(C.rows(), C.cols());
        Wc.setIdentity();

        As = Wm * A * Wm;
        Bs = Wm * B * Wc;
        BsT = Bs.transpose();
        Cs = Wc * C * Wc;

        Ss = Cs - BsT * Bs;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void SaddlePointSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        assert(rhs.cols() == 1);
        const Eigen::VectorXd Rm = rhs.block(0, 0, precond_num_, 1);
        const Eigen::VectorXd Rc = rhs.block(precond_num_, 0, rhs.size() - precond_num_, 1);

        const Eigen::VectorXd Rms = Wm * Rm;
        const Eigen::VectorXd Rcs = Wc * Rc;

        Eigen::VectorXd currentRms = Rms;
        Eigen::VectorXd currentRcs = Rcs;

        Eigen::VectorXd Rcst, Rmst;

        std::vector<Eigen::VectorXd> yu, yp, Rmu, Rmp, Rcu, Rcp;
        Eigen::VectorXd alphau;
        Eigen::VectorXd alphap;

        auto asymmetric_solver = LinearSolver::create(asymmetric_solver_name_, "");
        auto symmetric_solver = LinearSolver::create(symmetric_solver_name_, "");

        symmetric_solver->analyzePattern(Ss, Ss.rows() - 1);
        symmetric_solver->factorize(Ss);

        int i;
        for (i = 0; i < max_iter_; ++i)
        {
            yu.emplace_back(Rm.size());
            yp.emplace_back(Rc.size());

            yu[i].setZero();
            yp[i].setZero();

            //1
            // iters{i}.yu = gmres(As, iters{i}.Rms, iter_gmrs, eps_gm, outer_iter_gmrs);
            clock_t start_time = clock();
            asymmetric_solver->analyzePattern(As, As.rows());
            asymmetric_solver->factorize(As);
            assert(currentRms.size() == yu[i].size());
            asymmetric_solver->solve(currentRms, yu[i]);
            clock_t end_time = clock();
            std::cout << "Step 1 time cost = " << (end_time-start_time) << "ms" << std::endl;

            //2
            //Rcst = iters{i}.Rcs - Bs' * iters{i}.yu;
            Rcst = currentRcs - BsT * yu[i];

            //3
            //iters{i}.yp = bicgstab(Ss, Rcst, eps_cg, 10000);
            assert(Rcst.size() == yp[i].size());
            start_time = clock();
            symmetric_solver->solve(Rcst, yp[i]);
            end_time = clock();
            std::cout << "Step 2 time cost = " << (end_time-start_time) << "ms" << std::endl;

            //4
            //Rmst = iters{i}.Rms - Bs*iters{i}.yp;
            Rmst = currentRms - Bs * yp[i];

            //5
            // iters{i}.yu = gmres(As, Rmst, iter_gmrs, eps_gm, outer_iter_gmrs);
            assert(Rmst.size() == yu[i].size());
            yu[i].setZero();
            start_time = clock();
            asymmetric_solver->solve(Rmst, yu[i]);
            end_time = clock();
            std::cout << "Step 3 time cost = " << (end_time-start_time) << "ms" << std::endl;

            //update
            Rmu.emplace_back(As * yu[i]);
            Rmp.emplace_back(Bs * yp[i]);
            Rcu.emplace_back(BsT * yu[i]);
            Rcp.emplace_back(Cs * yp[i]);

            Eigen::MatrixXd Auu = Eigen::MatrixXd::Zero(i + 1, i + 1);
            Eigen::MatrixXd Aup = Eigen::MatrixXd::Zero(i + 1, i + 1);
            Eigen::MatrixXd Apu = Eigen::MatrixXd::Zero(i + 1, i + 1);
            Eigen::MatrixXd App = Eigen::MatrixXd::Zero(i + 1, i + 1);

            Eigen::VectorXd bu = Eigen::VectorXd::Zero(i + 1);
            Eigen::VectorXd bp = Eigen::VectorXd::Zero(i + 1);

            for (int k = 0; k <= i; ++k)
            {
                for (int j = 0; j <= i; ++j)
                {
                    Auu(k, j) = Rmu[k].dot(Rmu[j]) + Rcu[k].dot(Rcu[j]);
                    Aup(k, j) = Rmu[k].dot(Rmp[j]) + Rcu[k].dot(Rcp[j]);
                    Apu(k, j) = Rmp[k].dot(Rmu[j]) + Rcp[k].dot(Rcu[j]);
                    App(k, j) = Rmp[k].dot(Rmp[j]) + Rcp[k].dot(Rcp[j]);
                }

                bu(k) = Rms.dot(Rmu[k]) + Rcs.dot(Rcu[k]);
                bp(k) = Rms.dot(Rmp[k]) + Rcs.dot(Rcp[k]);
            }

            // Ao = [Auu Aup; Apu App];
            // bo = [bu; bp];
            Eigen::MatrixXd A(2 * (i + 1), 2 * (i + 1));
            Eigen::VectorXd b(2 * (i + 1));

            A.topLeftCorner(i + 1, i + 1) = Auu;
            A.topRightCorner(i + 1, i + 1) = Aup;
            A.bottomLeftCorner(i + 1, i + 1) = Apu;
            A.bottomRightCorner(i + 1, i + 1) = App;

            b.topRows(i + 1) = bu;
            b.bottomRows(i + 1) = bp;

            // alpha = A\b;
            Eigen::VectorXd alpha = A.ldlt().solve(b);
            // small_solver->analyzePattern(A, A.rows());
            // small_solver->solve(b, alpha);

            // alphau = alpha(1:i);
            // alphap = alpha(i+1:end);
            alphau = alpha.topRows(i + 1);
            alphap = alpha.bottomRows(i + 1);

            //TODO stopping condition!
            compute_solution(i + 1, alphau, alphap, yu, yp, Wm, Wc, result);
            final_res_norm_ = (Ain_ * result - rhs).norm();
            std::cout << "\nResidual = " << final_res_norm_ << "\n";

            if (final_res_norm_ < conv_tol_)
            {
                break;
            }

            // iters{i+1}.Rms = Rms;
            // iters{i+1}.Rcs = Rcs;
            currentRms = Rms;
            currentRcs = Rcs;

            for (int j = 0; j <= i; ++j)
            {
                // iters{i+1}.Rms = iters{i+1}.Rms - alphau(j)*iters{j}.Rmu - alphap(j)*iters{j}.Rmp;
                // iters{i+1}.Rcs = iters{i+1}.Rcs - alphau(j)*iters{j}.Rcu - alphap(j)*iters{j}.Rcp;
                currentRms -= alphau(j) * Rmu[j] + alphap(j) * Rmp[j];
                currentRcs -= alphau(j) * Rcu[j] + alphap(j) * Rcp[j];
            }
        }

        max_iter_ = i;
        // compute_solution(i, alphau, alphap, yu, yp, Wm, Wc, result);
    }

    ////////////////////////////////////////////////////////////////////////////////

    SaddlePointSolver::~SaddlePointSolver()
    {
    }

} // namespace polysolve
