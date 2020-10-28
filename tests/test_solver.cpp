////////////////////////////////////////////////////////////////////////////////
#include <polysolve/FEMSolver.hpp>

#include <catch.hpp>
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
        INFO("solver: " + s);
        REQUIRE(err < 1e-8);
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
    const std::string path = POLYSOLVE_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    bool ok = loadMarket(A, path + "/A0.mat");
    REQUIRE(ok);

    Eigen::VectorXd b;
    ok = loadMarketVector(b, path + "/b0.mat");
    REQUIRE(ok);

    auto solver = LinearSolver::create("SaddlePointSolver", "");
    solver->analyzePattern(A, 9934);
    Eigen::VectorXd x(A.rows());
    solver->solve(b, x);
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}