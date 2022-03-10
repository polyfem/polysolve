////////////////////////////////////////////////////////////////////////////////
#include <polysolve/FEMSolver.hpp>
#include <polysolve/CHOLMODSolver.hpp>

#include <catch2/catch.hpp>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
////////////////////////////////////////////////////////////////////////////////

using namespace polysolve;

TEST_CASE("all", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solvers = LinearSolver::availableSolvers();

    for (const auto &s : solvers)
    {
        if (s == "Eigen::DGMRES")
            continue;
#ifdef WIN32
        if (s == "Eigen::ConjugateGradient" || s == "Eigen::BiCGSTAB" || s == "Eigen::GMRES" || s == "Eigen::MINRES")
            continue;
#endif
        auto solver = LinearSolver::create(s, "");
        solver->setParameters(R"({"conv_tol": 1e-10})"_json);
        Eigen::VectorXd b(A.rows());
        b.setRandom();
        Eigen::VectorXd x(b.size());
        x.setZero();

        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);

        // solver->getInfo(solver_info);

        // std::cout<<"Solver error: "<<x<<std::endl;
        const double err = (A * x - b).norm();
        INFO("solver: " + s);
        REQUIRE(err < 1e-8);
    }
}

TEST_CASE("pre_factor", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solvers = LinearSolver::availableSolvers();

    for (const auto &s : solvers)
    {
        if (s == "Eigen::DGMRES")
            continue;
#ifdef WIN32
        if (s == "Eigen::ConjugateGradient" || s == "Eigen::BiCGSTAB" || s == "Eigen::GMRES" || s == "Eigen::MINRES")
            continue;
#endif
        auto solver = LinearSolver::create(s, "");
        solver->analyzePattern(A, A.rows());

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

            // solver->getInfo(solver_info);

            // std::cout<<"Solver error: "<<x<<std::endl;
            const double err = (Atmp * x - b).norm();
            INFO("solver: " + s);
            REQUIRE(err < 1e-8);
        }
    }
}

#ifdef POLYSOLVE_WITH_HYPRE
TEST_CASE("hypre", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solver = LinearSolver::create("Hypre", "");
    // solver->setParameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyzePattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);

    // solver->getInfo(solver_info);

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("hypre_initial_guess", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    // solver->setParameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        json solver_info;

        auto solver = LinearSolver::create("Hypre", "");
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);
        solver->getInfo(solver_info);

        REQUIRE(solver_info["num_iterations"] > 1);
    }

    {
        json solver_info;
        auto solver = LinearSolver::create("Hypre", "");
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);

        solver->getInfo(solver_info);

        REQUIRE(solver_info["num_iterations"] == 1);
    }

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}
#endif

#ifdef POLYSOLVE_WITH_AMGCL
TEST_CASE("amgcl_initial_guess", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    // solver->setParameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        json solver_info;

        auto solver = LinearSolver::create("AMGCL", "");
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);
        solver->getInfo(solver_info);

        REQUIRE(solver_info["num_iterations"] > 0);
    }

    {
        json solver_info;
        auto solver = LinearSolver::create("AMGCL", "");
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);

        solver->getInfo(solver_info);

        REQUIRE(solver_info["num_iterations"] == 0);
    }

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}
#endif

TEST_CASE("saddle_point_test", "[solver]")
{
#ifdef WIN32
#ifndef NDEBUG
    return;
#endif
#endif
    const std::string path = POLYSOLVE_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    bool ok = loadMarket(A, path + "/A0.mat");
    REQUIRE(ok);

    Eigen::VectorXd b;
    ok = loadMarketVector(b, path + "/b0.mat");
    REQUIRE(ok);

    auto solver = LinearSolver::create("SaddlePointSolver", "");
    solver->analyzePattern(A, 9934);
    solver->factorize(A);
    Eigen::VectorXd x(A.rows());
    solver->solve(b, x);
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("CHOLMOD", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;
    Eigen::SparseMatrix<double, Eigen::ColMajor, long int> A;
    bool ok = loadMarket(A, path + "/nd6k.mtx");
    REQUIRE(ok);

    Eigen::VectorXd b(A.rows());
    b.setOnes();
    Eigen::VectorXd x(A.rows());
    x.setZero();

    CHOLMODSolver solver;
    std::cout<<"here1\n";
    solver.analyzePattern(A);
    std::cout<<"here2\n";
    solver.factorize(A);
    std::cout<<"here3\n";
    solver.solve(b, x);
    std::cout<<"here4\n";
    
    const double err = (b - (A * x)).norm();
    REQUIRE(err < 1e-8);
}
