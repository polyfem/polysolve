#include <polysolve/FEMSolver.hpp>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <fstream>
#include <vector>
#include <ctime>
#include <polysolve/LinearSolverAMGCL_cuda.hpp>
#include <cuda_runtime_api.h>
//////////////////////////////////////////////////////////////////////////
#include <catch2/catch.hpp>

using namespace polysolve;

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

TEST_CASE("PETSC-STRUMPACK", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/crystm03.mtx");
    REQUIRE(ok);

    auto solver = LinearSolver::create("PETSC_Solver", "");
    // solver->setParameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyzePattern(A, A.rows());
    solver->factorize(A, 0, 5);
    solver->solve(b, x);

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("amgcl_crystm03_cg", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;
    std::string MatrixName = "crystm03.mtx";
    Eigen::SparseMatrix<double> A;
    loadSymmetric(A, path + "/" + MatrixName);

    std::cout << "Matrix Load OK" << std::endl;

    Eigen::VectorXd b(A.rows());
    b.setOnes();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        amgcl::profiler<> prof("crystm03_GPU");
        json solver_info;
        auto solver = LinearSolver::create("AMGCL_cuda", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 10000;
        params["AMGCL"]["solver_type"] = "cg";
        solver->setParameters(params);
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x);
        prof.toc("solve");
        solver->getInfo(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << prof << std::endl;
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);
}

TEST_CASE("amgcl_crystm03_bicgstab", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;
    std::string MatrixName = "crystm03.mtx";
    Eigen::SparseMatrix<double> A;
    loadSymmetric(A, path + "/" + MatrixName);

    std::cout << "Matrix Load OK" << std::endl;

    Eigen::VectorXd b(A.rows());
    b.setOnes();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        amgcl::profiler<> prof("crystm03_GPU");
        json solver_info;
        auto solver = LinearSolver::create("AMGCL_cuda", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 10000;
        params["AMGCL"]["solver_type"] = "bicgstab";
        solver->setParameters(params);
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x);
        prof.toc("solve");
        solver->getInfo(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << prof << std::endl;
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);
}

TEST_CASE("cusolverdn", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solver = LinearSolver::create("cuSolverDN", "");
    // solver->setParameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyzePattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}
TEST_CASE("cusolverdn_dense", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;

    Eigen::MatrixXd A(4, 4);
    for (int i = 0; i < 4; i++)
    {
        A(i, i) = 1.0;
    }
    A(0, 1) = 1.0;
    A(3, 0) = 1.0;

    auto solver = LinearSolver::create("cuSolverDN", "");
    // solver->setParameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyzePattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("cusolverdn_5cubes", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;
    auto solver = LinearSolver::create("cuSolverDN", "");

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

        solver->analyzePattern(A, A.rows());

        // std::chrono::steady_clock::time_point beginf = std::chrono::steady_clock::now();
        solver->factorize(A);
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