////////////////////////////////////////////////////////////////////////////////
#include <polyfem-solvers/FEMSolver.hpp>

#include <catch.hpp>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
////////////////////////////////////////////////////////////////////////////////

using namespace polyfem;

TEST_CASE("all", "[solver]")
{
    const std::string path = POLYFEM_SOLVERS_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solvers = LinearSolver::availableSolvers();

    for (const auto &s : solvers)
    {
        if (s == "Eigen::DGMRES")
            continue;
        auto solver = LinearSolver::create(s, "");
        // solver->setParameters(params);
        Eigen::VectorXd b(A.rows());
        b.setRandom();
        Eigen::VectorXd x(b.size());

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

#ifdef POLYFEM_SOLVERS_WITH_HYPRE
TEST_CASE("hypre", "[solver]")
{
    const std::string path = POLYFEM_SOLVERS_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solver = LinearSolver::create("Hypre", "");
    // solver->setParameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());

    solver->analyzePattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);

    // solver->getInfo(solver_info);

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}
#endif
