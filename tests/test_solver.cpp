//////////////////////////////////////////////////////////////////////////
#include <polysolve/FEMSolver.hpp>
#include <catch2/catch.hpp>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <fstream>
#include <vector>
#include <ctime>
#include <polysolve/LinearSolverAMGCL.hpp>
//////////////////////////////////////////////////////////////////////////

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
        json params;
        params[s]["tolerance"] = 1e-10;
        solver->setParameters(params);
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

TEST_CASE("eigen_params", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solvers = LinearSolver::availableSolvers();

    for (const auto &s : solvers)
    {
        if (s == "Eigen::ConjugateGradient" || s == "Eigen::BiCGSTAB" || s == "Eigen::GMRES" || s == "Eigen::MINRES" || s == "Eigen::LeastSquaresConjugateGradient" || s == "Eigen::DGMRES")
        {
            auto solver = LinearSolver::create(s, "");
            json params;
            params[s]["max_iter"] = 1000;
            params[s]["tolerance"] = 1e-10;
            solver->setParameters(params);

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

#ifdef POLYSOLVE_WITH_AMGCL
TEST_CASE("amgcl_blocksolver_small_scale", "[solver]")
{
#ifndef NDEBUG
    return;
#endif
    const std::string path = POLYSOLVE_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    // solver->setParameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(A.rows());
    Eigen::VectorXd x_b(A.rows());
    x.setZero();
    x_b.setZero();
    {
        json solver_info;

        auto solver = LinearSolver::create("AMGCL", "");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        solver->setParameters(params);
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);
        solver->getInfo(solver_info);

        REQUIRE(solver_info["num_iterations"] > 0);
        const double err = (A * x - b).norm();
        REQUIRE(err < 1e-5);
    }

    {
        json solver_info;
        auto solver = LinearSolver::create("AMGCL", "");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["block_size"] = 3;
        solver->setParameters(params);
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x_b);
        solver->getInfo(solver_info);

        REQUIRE(solver_info["num_iterations"] > 0);
        const double err = (A * x_b - b).norm();
        REQUIRE(err < 1e-5);
    }
}
#endif

#ifdef POLYSOLVE_WITH_AMGCL
TEST_CASE("amgcl_blocksolver_b2", "[solver]")
{
#ifndef NDEBUG
    return;
#endif
    const std::string path = POLYSOLVE_DATA_DIR;
    std::string MatrixName = "gr_30_30.mtx";
    Eigen::SparseMatrix<double> A;
    loadSymmetric(A, path + "/" + MatrixName);

    std::cout << "Matrix Load OK" << std::endl;

    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(A.rows());
    Eigen::VectorXd x_b(A.rows());
    x.setOnes();
    x_b.setOnes();
    {
        amgcl::profiler<> prof("gr_30_30_Scalar");
        json solver_info;
        auto solver = LinearSolver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 1000;
        solver->setParameters(params);
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x);
        prof.toc("solve");
        solver->getInfo(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    {
        amgcl::profiler<> prof("gr_30_30_Block");
        json solver_info;
        auto solver = LinearSolver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 1000;
        params["AMGCL"]["block_size"] = 2;
        solver->setParameters(params);
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x_b);
        prof.toc("solve");
        solver->getInfo(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);
    REQUIRE((A * x_b - b).norm() / b.norm() < 1e-7);
}
#endif

#ifdef POLYSOLVE_WITH_AMGCL
TEST_CASE("amgcl_blocksolver_crystm03_CG", "[solver]")
{
#ifndef NDEBUG
    return;
#endif
    std::cout << "Polysolve AMGCL Solver" << std::endl;
    const std::string path = POLYSOLVE_DATA_DIR;
    std::string MatrixName = "crystm03.mtx";
    Eigen::SparseMatrix<double> A;
    loadSymmetric(A, path + "/" + MatrixName);
    std::cout << "Matrix Load OK" << std::endl;
    Eigen::VectorXd b(A.rows());
    b.setOnes();
    Eigen::VectorXd x_b(A.rows());
    x_b.setZero();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        amgcl::profiler<> prof("crystm03_Block");
        json solver_info;
        auto solver = LinearSolver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 1000;
        params["AMGCL"]["block_size"] = 3;
        solver->setParameters(params);
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x_b);
        prof.toc("solve");
        solver->getInfo(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    {
        amgcl::profiler<> prof("crystm03_Scalar");
        json solver_info;
        auto solver = LinearSolver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 10000;
        solver->setParameters(params);
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x);
        prof.toc("solve");
        solver->getInfo(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);
    REQUIRE((A * x_b - b).norm() / b.norm() < 1e-7);
}
#endif

#ifdef POLYSOLVE_WITH_AMGCL
TEST_CASE("amgcl_blocksolver_crystm03_Bicgstab", "[solver]")
{
#ifndef NDEBUG
    return;
#endif
    std::cout << "Polysolve AMGCL Solver" << std::endl;
    const std::string path = POLYSOLVE_DATA_DIR;
    std::string MatrixName = "crystm03.mtx";
    Eigen::SparseMatrix<double> A;
    loadSymmetric(A, path + "/" + MatrixName);

    std::cout << "Matrix Load OK" << std::endl;

    Eigen::VectorXd b(A.rows());
    b.setOnes();
    Eigen::VectorXd x_b(A.rows());
    x_b.setZero();
    Eigen::VectorXd x(A.rows());
    x.setZero();
    {
        amgcl::profiler<> prof("crystm03_Block");
        json solver_info;
        auto solver = LinearSolver::create("AMGCL", "");
        prof.tic("setup");
        json params;
        params["AMGCL"]["tolerance"] = 1e-8;
        params["AMGCL"]["max_iter"] = 10000;
        params["AMGCL"]["block_size"] = 3;
        params["AMGCL"]["solver_type"] = "bicgstab";
        solver->setParameters(params);
        solver->analyzePattern(A, A.rows());
        solver->factorize(A);
        prof.toc("setup");
        prof.tic("solve");
        solver->solve(b, x_b);
        prof.toc("solve");
        solver->getInfo(solver_info);
        REQUIRE(solver_info["num_iterations"] > 0);
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    {
        amgcl::profiler<> prof("crystm03_Scalar");
        json solver_info;
        auto solver = LinearSolver::create("AMGCL", "");
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
        std::cout << solver_info["num_iterations"] << std::endl;
        std::cout << solver_info["final_res_norm"] << std::endl
                  << prof << std::endl;
    }
    REQUIRE((A * x - b).norm() / b.norm() < 1e-7);
    REQUIRE((A * x_b - b).norm() / b.norm() < 1e-7);
}
#endif

#ifdef POLYSOLVE_WITH_CUSOLVER
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

    solver->analyzePattern_dense(A, A.rows());
    solver->factorize_dense(A);
    solver->solve(b, x);

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-8);
}

TEST_CASE("cusolverdn_dense_float", "[solver]")
{
    const std::string path = POLYSOLVE_DATA_DIR;

    Eigen::MatrixXd A(4, 4);
    for (int i = 0; i < 4; i++)
    {
        A(i, i) = 1.0;
    }
    A(0, 1) = 1.0;
    A(3, 0) = 1.0;

    auto solver = LinearSolver::create("cuSolverDN_float", "");
    // solver->setParameters(params);
    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x(b.size());
    x.setZero();

    solver->analyzePattern_dense(A, A.rows());
    solver->factorize_dense(A);
    solver->solve(b, x);

    // std::cout<<"Solver error: "<<x<<std::endl;
    const double err = (A * x - b).norm();
    REQUIRE(err < 1e-6);
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

        solver->analyzePattern_dense(A, A.rows());

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
#endif