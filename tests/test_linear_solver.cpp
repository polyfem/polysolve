//////////////////////////////////////////////////////////////////////////
#include <polysolve/Types.hpp>
#include <polysolve/linear/FEMSolver.hpp>
#include <polysolve/linear/SPQR.hpp>

#include <polysolve/Utils.hpp>

#ifdef POLYSOLVE_WITH_AMGCL
#include <polysolve/linear/AMGCL.hpp>
#endif

#include <spdlog/sinks/stdout_color_sinks.h>

#include <catch2/catch.hpp>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>
//////////////////////////////////////////////////////////////////////////

using namespace polysolve;
using namespace polysolve::linear;

void loadSymmetric(Eigen::SparseMatrix<double> &A, std::string PATH)
{
    std::ifstream fin(PATH);
    long int M, N, L;
    while (fin.peek() == '%')
    {
        fin.ignore(2048, '\n');
    }
    fin >> M >> N >> L;
    A.resize(M, N);
    A.reserve(L * 2 - M);
    std::vector<Eigen::Triplet<double>> triple;
    for (size_t i = 0; i < L; i++)
    {
        int m, n;
        double data;
        fin >> m >> n >> data;
        triple.push_back(Eigen::Triplet<double>(m - 1, n - 1, data));
        if (m != n)
        {
            triple.push_back(Eigen::Triplet<double>(n - 1, m - 1, data));
        }
    }
    fin.close();
    A.setFromTriplets(triple.begin(), triple.end());
};

TEST_CASE("jse", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger");
    logger->set_level(spdlog::level::warn);

    json input = {};
    auto solver = Solver::create(input, *logger);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    const double err = (A * x - b).norm();
    INFO("solver: " + solver->name());
    REQUIRE(err < 1e-8);
}

TEST_CASE("multi-solver", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger");
    logger->set_level(spdlog::level::warn);

    json input = {};
    input["solver"] = {"Hypre", "Eigen::SimplicialLDLT"};
    auto solver = Solver::create(input, *logger);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    const double err = (A * x - b).norm();
    INFO("solver: " + solver->name());
    REQUIRE(err < 1e-8);
}

TEST_CASE("all", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);
    json solver_info;
    Eigen::MatrixXd A_dense(A);

    auto solvers = Solver::available_solvers();
    for (const auto &s : solvers)
    {
        std::cout << s << std::endl;
    }

    for (const auto &s : solvers)
    {
        if (s == "Eigen::DGMRES")
            continue;
#ifdef WIN32
        if (s == "Eigen::ConjugateGradient" || s == "Eigen::BiCGSTAB" || s == "Eigen::GMRES" || s == "Eigen::MINRES")
            continue;
#endif
        auto solver = Solver::create(s, "");
        json params;
        params[s]["tolerance"] = 1e-10;
        solver->set_parameters(params);
        Eigen::VectorXd b(A.rows());
        b.setRandom();
        Eigen::VectorXd x(b.size());
        x.setZero();

        if (solver->is_dense())
        {
            solver->analyze_pattern_dense(A, A.rows());
            solver->factorize_dense(A);
        }
        else
        {
            solver->analyze_pattern(A, A.rows());
            solver->factorize(A);
        }

        solver->solve(b, x);

        REQUIRE(solver->name() == s);

        solver->get_info(solver_info);

        // std::cout<<"Solver error: "<<x<<std::endl;
        const double err = (A * x - b).norm();
        INFO("solver: " + s);
        REQUIRE(err < 1e-8);
    }
}

TEST_CASE("eigen_params", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solvers = Solver::available_solvers();

    for (const auto &s : solvers)
    {
        if (s == "Eigen::ConjugateGradient" || s == "Eigen::BiCGSTAB" || s == "Eigen::GMRES" || s == "Eigen::MINRES" || s == "Eigen::LeastSquaresConjugateGradient" || s == "Eigen::DGMRES")
        {
            auto solver = Solver::create(s, "");
            json params;
            params[s]["max_iter"] = 1000;
            params[s]["tolerance"] = 1e-10;
            solver->set_parameters(params);

            Eigen::VectorXd b(A.rows());
            b.setRandom();
            Eigen::VectorXd x(b.size());
            x.setZero();

            solver->analyze_pattern(A, A.rows());
            solver->factorize(A);
            solver->solve(b, x);

            // solver->get_info(solver_info);

            // std::cout<<"Solver error: "<<x<<std::endl;
            const double err = (A * x - b).norm();
            INFO("solver: " + s);
            REQUIRE(err < 1e-8);
        }
    }
}

TEST_CASE("pre_factor", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solvers = Solver::available_solvers();

    for (const auto &s : solvers)
    {
        if (s == "Eigen::DGMRES")
            continue;
#ifdef WIN32
        if (s == "Eigen::ConjugateGradient" || s == "Eigen::BiCGSTAB" || s == "Eigen::GMRES" || s == "Eigen::MINRES")
            continue;
#endif
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        auto solver = Solver::create(s, "");
        solver->analyze_pattern(A, A.rows());

        std::default_random_engine eng{42};
        std::uniform_real_distribution<double> urd(0.1, 5);

        for (int i = 0; i < 10; ++i)
        {
            std::vector<Eigen::Triplet<double>> tripletList;

            for (int k = 0; k < A.outerSize(); ++k)
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
                {
                    if (it.row() == it.col())
                    {
                        tripletList.emplace_back(it.row(), it.col(), urd(eng) * 100);
                    }
                    else if (it.row() < it.col())
                    {
                        const double val = -urd(eng);
                        tripletList.emplace_back(it.row(), it.col(), val);
                        tripletList.emplace_back(it.col(), it.row(), val);
                    }
                }
            }

            Eigen::SparseMatrix<double> Atmp(A.rows(), A.cols());
            Atmp.setFromTriplets(tripletList.begin(), tripletList.end());

            Eigen::VectorXd b(Atmp.rows());
            b.setRandom();
            Eigen::VectorXd x(b.size());
            x.setZero();

            solver->factorize(Atmp);
            solver->solve(b, x);

            // solver->get_info(solver_info);

            // std::cout<<"Solver error: "<<x<<std::endl;
            const double err = (Atmp * x - b).norm();
            INFO("solver: " + s);
            REQUIRE(err < 1e-8);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << s << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    }
}

TEST_CASE("hypre", "[solver]")
{
    std::unique_ptr<Solver> solver;

    try
    {
        solver = Solver::create("Hypre", "");
    }
    catch (const std::exception &)
    {
        return;
    }
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    // solver->set_parameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);

    // solver->get_info(solver_info);

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("hypre_initial_guess", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    // solver->set_parameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        json solver_info;
        std::unique_ptr<Solver> solver;
        try
        {
            solver = Solver::create("Hypre", "");
        }
        catch (const std::exception &)
        {
            return;
        }
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);
        solver->get_info(solver_info);

        REQUIRE(solver_info["num_iterations"] > 1);
    }

    {
        json solver_info;
        std::unique_ptr<Solver> solver;

        try
        {
            solver = Solver::create("Hypre", "");
        }
        catch (const std::exception &)
        {
            return;
        }
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);

        solver->get_info(solver_info);

        REQUIRE(solver_info["num_iterations"] == 1);
    }

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("amgcl_initial_guess", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    // solver->set_parameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        json solver_info;
        std::unique_ptr<Solver> solver;

        try
        {
            solver = Solver::create("AMGCL", "");
        }
        catch (const std::exception &)
        {
            return;
        }
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);
        solver->get_info(solver_info);

        REQUIRE(solver_info["num_iterations"] > 0);
    }

    {
        json solver_info;
        std::unique_ptr<Solver> solver;
        try
        {
            solver = Solver::create("AMGCL", "");
        }
        catch (const std::exception &)
        {
            return;
        }
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);

        solver->get_info(solver_info);

        REQUIRE(solver_info["num_iterations"] == 0);
    }

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("saddle_point_test", "[solver]")
{
#ifdef WIN32
#ifndef NDEBUG
    return;
#endif
#endif
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    bool ok = loadMarket(A, path + "/A0.mat");
    REQUIRE(ok);

    Eigen::VectorXd b;
    ok = loadMarketVector(b, path + "/b0.mat");
    REQUIRE(ok);

    auto solver = Solver::create("SaddlePointSolver", "");
    solver->analyze_pattern(A, 9934);
    solver->factorize(A);
    Eigen::VectorXd x(A.rows());
    solver->solve(b, x);

    json solver_info;
    solver->get_info(solver_info);

    REQUIRE(solver->name() == "SaddlePointSolver");

    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

#ifdef POLYSOLVE_WITH_AMGCL
TEST_CASE("amgcl_blocksolver_small_scale", "[solver]")
{
#ifndef NDEBUG
    return;
#endif
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    // solver->set_parameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(A.rows());
    Eigen::VectorXd x_b(A.rows());
    x.setZero();
    x_b.setZero();
    {
        json solver_info;

        auto solver = Solver::create("AMGCL", "");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);
        solver->get_info(solver_info);

        REQUIRE(solver_info["num_iterations"] > 0);
        const double err = (A * x - b).norm();
        REQUIRE(err < 1e-5);
    }

    {
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["block_size"] = 3;
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x_b);
        solver->get_info(solver_info);

        REQUIRE(solver_info["num_iterations"] > 0);
        const double err = (A * x_b - b).norm();
        REQUIRE(err < 1e-5);
    }
}

TEST_CASE("amgcl_blocksolver_b2", "[solver]")
{
#ifndef NDEBUG
    return;
#endif
    const std::string path = POLYFEM_DATA_DIR;
    std::string MatrixName = "gr_30_30.mtx";
    Eigen::SparseMatrix<double> A;
    loadSymmetric(A, path + "/" + MatrixName);

    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(A.rows());
    Eigen::VectorXd x_b(A.rows());
    x.setOnes();
    x_b.setOnes();
    {
        amgcl::profiler<> prof("gr_30_30_Scalar");
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 1000;
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x);
        prof.toc("solve");
        solver->get_info(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    {
        amgcl::profiler<> prof("gr_30_30_Block");
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 1000;
        params["AMGCL"]["block_size"] = 2;
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x_b);
        prof.toc("solve");
        solver->get_info(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);
    REQUIRE((A * x_b - b).norm() / b.norm() < 1e-7);
}

TEST_CASE("amgcl_blocksolver_crystm03_CG", "[solver]")
{
#ifndef NDEBUG
    return;
#endif
    std::cout << "Polysolve AMGCL Solver" << std::endl;
    const std::string path = POLYFEM_DATA_DIR;
    std::string MatrixName = "crystm03.mtx";
    Eigen::SparseMatrix<double> A;
    loadSymmetric(A, path + "/" + MatrixName);
    Eigen::VectorXd b(A.rows());
    b.setOnes();
    Eigen::VectorXd x_b(A.rows());
    x_b.setZero();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        amgcl::profiler<> prof("crystm03_Block");
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 1000;
        params["AMGCL"]["block_size"] = 3;
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x_b);
        prof.toc("solve");
        solver->get_info(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    {
        amgcl::profiler<> prof("crystm03_Scalar");
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 10000;
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x);
        prof.toc("solve");
        solver->get_info(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);
    REQUIRE((A * x_b - b).norm() / b.norm() < 1e-7);
}

TEST_CASE("amgcl_blocksolver_crystm03_Bicgstab", "[solver]")
{
#ifndef NDEBUG
    return;
#endif
    std::cout << "Polysolve AMGCL Solver" << std::endl;
    const std::string path = POLYFEM_DATA_DIR;
    std::string MatrixName = "crystm03.mtx";
    Eigen::SparseMatrix<double> A;
    loadSymmetric(A, path + "/" + MatrixName);

    Eigen::VectorXd b(A.rows());
    b.setOnes();
    Eigen::VectorXd x_b(A.rows());
    x_b.setZero();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        amgcl::profiler<> prof("crystm03_Block");
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 10000;
        params["AMGCL"]["block_size"] = 3;
        params["AMGCL"]["solver_type"] = "bicgstab";
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x_b);
        prof.toc("solve");
        solver->get_info(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    {
        amgcl::profiler<> prof("crystm03_Scalar");
        json solver_info;
        auto solver = Solver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 10000;
        params["AMGCL"]["solver_type"] = "bicgstab";
        solver->set_parameters(params);
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x);
        prof.toc("solve");
        solver->get_info(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);
    REQUIRE((A * x_b - b).norm() / b.norm() < 1e-7);
}
#endif

TEST_CASE("cusolverdn", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);
    std::unique_ptr<Solver> solver;
    try
    {
        solver = Solver::create("cuSolverDN", "");
    }
    catch (const std::exception &)
    {
        return;
    }
    // solver->set_parameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("cusolverdn_dense", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;

    Eigen::MatrixXd A(4, 4);
    for (int i = 0; i < 4; i++)
    {
        A(i, i) = 1.0;
    }
    A(0, 1) = 1.0;
    A(3, 0) = 1.0;
    std::unique_ptr<Solver> solver;
    try
    {
        solver = Solver::create("cuSolverDN", "");
    }
    catch (const std::exception &)
    {
        return;
    }
    // solver->set_parameters(params);
    for (int i = 0; i < 5; ++i)
    {
        Eigen::VectorXd b(A.rows());
        b.setRandom();
        Eigen::VectorXd x(b.size());
        x.setZero();

        solver->analyze_pattern_dense(A, A.rows());
        solver->factorize_dense(A);
        solver->solve(b, x);

        // std::cout<<"Solver error: "<<x<<std::endl;
        const double err = (A * x - b).norm();
        REQUIRE(err < 1e-8);
    }
}

TEST_CASE("cusolverdn_dense_float", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;

    Eigen::MatrixXd A(4, 4);
    for (int i = 0; i < 4; i++)
    {
        A(i, i) = 1.0;
    }
    A(0, 1) = 1.0;
    A(3, 0) = 1.0;
    std::unique_ptr<Solver> solver;
    try
    {
        solver = Solver::create("cuSolverDN_float", "");
    }
    catch (const std::exception &)
    {
        return;
    }
    // solver->set_parameters(params);

    for (int i = 0; i < 5; ++i)
    {
        Eigen::VectorXd b(A.rows());
        b.setRandom();
        Eigen::VectorXd x(b.size());
        x.setZero();

        solver->analyze_pattern_dense(A, A.rows());
        solver->factorize_dense(A);
        solver->solve(b, x);

        // std::cout<<"Solver error: "<<x<<std::endl;
        const double err = (A * x - b).norm();
        REQUIRE(err < 1e-6);
    }
}

TEST_CASE("cusolverdn_5cubes", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;

    std::unique_ptr<Solver> solver;
    try
    {
        solver = Solver::create("cuSolverDN", "");
    }
    catch (const std::exception &)
    {
        return;
    }

    // std::ofstream factorize_times_file(path+"/factorize_times_5cubes.txt");
    // std::ofstream solve_times_file(path+"/solve_times_5cubes.txt");

    for (int i = 0; i <= 1091; i++)
    {
        Eigen::MatrixXd A(120, 120);
        std::string hessian_path = path + "/matrixdata-5cubes/hessian" + std::to_string(i) + ".txt";
        std::ifstream hessian_file(hessian_path);
        for (int m = 0; m < 120; m++)
        {
            for (int n = 0; n < 120; n++)
            {
                hessian_file >> A(m, n);
            }
        }

        Eigen::VectorXd b(A.rows());
        std::string gradient_path = path + "/matrixdata-5cubes/gradient" + std::to_string(i) + ".txt";
        std::ifstream gradient_file(gradient_path);
        for (int m = 0; m < 120; m++)
        {
            gradient_file >> b(m);
        }

        Eigen::VectorXd x(b.size());
        x.setZero();

        solver->analyze_pattern_dense(A, A.rows());

        // std::chrono::steady_clock::time_point beginf = std::chrono::steady_clock::now();
        solver->factorize_dense(A);
        // std::chrono::steady_clock::time_point endf = std::chrono::steady_clock::now();
        // std::cout << "time to factorize: " << std::chrono::duration_cast<std::chrono::nanoseconds>(endf-beginf).count() << std::endl;
        // factorize_times_file << std::chrono::duration_cast<std::chrono::nanoseconds>(endf-beginf).count() << " ";

        // std::chrono::steady_clock::time_point begins = std::chrono::steady_clock::now();
        solver->solve(b, x);
        // std::chrono::steady_clock::time_point ends = std::chrono::steady_clock::now();
        // std::cout << "time to solve: " << std::chrono::duration_cast<std::chrono::nanoseconds>(ends-begins).count() << std::endl;
        // solve_times_file << std::chrono::duration_cast<std::chrono::nanoseconds>(ends-begins).count() << " ";

        // std::cout << "Ax norm: " << (A*x).norm() << std::endl;
        // std::cout << "b norm: " << b.norm() << std::endl;

        const double err = (A * x - b).norm();
        REQUIRE(err < 1e-8);
    }
}

TEST_CASE("spqr_sparse_product", "[solver]")
{

    Eigen::MatrixXd A(4, 4);
    for (int i = 0; i < 4; i++)
    {
        A(i, i) = 1.0;
    }
    A(0, 1) = 1.0;
    A(3, 0) = 1.0;
    std::unique_ptr<Solver> solver;
    try
    {
        solver = Solver::create("Eigen::SPQR", "");
    }
    catch (const std::exception &)
    {
        return;
    }

    using Type = EigenDirect<Eigen::SPQR<polysolve::StiffnessMatrix>>;

    Type *typed_solver = dynamic_cast<Type *>(solver.get());
    REQUIRE(typed_solver != nullptr);
    // solver->set_parameters(params);

    for (int i = 0; i < 5; ++i)
    {
        A = Eigen::MatrixXd::Random(5, 5);
        auto As = A.sparseView().eval();

        // do a qr so i have a q
        Eigen::SPQR<StiffnessMatrix> spqr(As);
        auto Q = spqr.matrixQ();

        // get a random matrix to multiply against
        Eigen::MatrixXd B(A.rows(), 5);
        B.setRandom();
        Eigen::MatrixXd dense = Q * B;

        // make a sparse version
        auto Bs = B.sparseView().eval();
        StiffnessMatrix sparse = Q * Bs;

        // check that the result of the product is the same
        CHECK((dense - sparse).norm() < 1e-10);

        // try to extract the Q matrix as a dense matrix
        Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(A.rows(), A.rows());
        Eigen::MatrixXd denseQ = Q * Id;

        // use the product with B to get a weak equivalence once again
        Eigen::MatrixXd dense2 = denseQ * B;
        CHECK((dense2 - dense).norm() < 1e-10);

        // now try using a sparse product
        StiffnessMatrix I(A.rows(), A.rows());
        I.setIdentity();
        StiffnessMatrix myQ = Q * I;
        StiffnessMatrix sparse2 = myQ * Bs;
        CHECK((sparse2 - sparse).norm() < 1e-10);
    }
}
