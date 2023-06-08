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

TEST_CASE("amgcl_crystm03_cg", "[solver]"){
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
        std::cout<< prof << std::endl;    
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);    
}

TEST_CASE("amgcl_crystm03_bicgstab", "[solver]"){
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
        std::cout<< prof << std::endl;    
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);    
}
