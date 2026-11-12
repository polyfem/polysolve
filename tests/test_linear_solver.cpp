//////////////////////////////////////////////////////////////////////////
#include <polysolve/Types.hpp>
#include <polysolve/linear/FEMSolver.hpp>

#include <polysolve/Utils.hpp>

#ifdef POLYSOLVE_WITH_AMGCL
#include <polysolve/linear/AMGCL.hpp>
#endif

#include <spdlog/sinks/stdout_color_sinks.h>

#include <catch2/catch.hpp>
#include <algorithm>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
//////////////////////////////////////////////////////////////////////////

using namespace polysolve;
using namespace polysolve::linear;

#if defined(POLYSOLVE_WITH_MPI) && !defined(_WIN32)
#define POLYSOLVE_TEST_RANKS
namespace
{
    // Sets NANOMPI_NUM_RANKS, the number of nano-mpi ranks a solver starts,
    // while it is in scope. It only affects ranks started after it is set.
    class ScopedNumRanks
    {
    public:
        explicit ScopedNumRanks(const int n)
        {
            if (const char *old = std::getenv("NANOMPI_NUM_RANKS"))
            {
                had_old_ = true;
                old_ = old;
            }
            setenv("NANOMPI_NUM_RANKS", std::to_string(n).c_str(), 1);
        }

        ~ScopedNumRanks()
        {
            if (had_old_)
                setenv("NANOMPI_NUM_RANKS", old_.c_str(), 1);
            else
                unsetenv("NANOMPI_NUM_RANKS");
        }

    private:
        bool had_old_ = false;
        std::string old_;
    };
} // namespace
#endif

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

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger_jse");
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

    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger_multi");
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
        if (s == "MAS")
        {
            params[s]["relative_tolerance"] = 0.0;
            params[s]["absolute_tolerance"] = 1e-8;
            params[s]["use_preconditioned_residual_norm"] = false;
            solver->set_parameters(params);
        }
        else if (s == "GPUHybrid" || s == "CPUHybrid")
        {
            params[s]["relative_tolerance"] = 0.0;
            params[s]["absolute_tolerance"] = 1e-8;
            solver->set_parameters(params);
        }
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

// Test block dim 1, 2, 3
TEST_CASE("mas_block_dim", "[.][solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
    // const bool ok = loadMarket(A, path + "/A_2.mat");
    bool ok = loadMarket(A, "/home/ian/local_code/polysolve/some_ls_data/1e8/frame_00040_ns_0_A.mtx");
    REQUIRE(ok);
    ok = Eigen::loadMarketVector(b, "/home/ian/local_code/polysolve/some_ls_data/1e8/frame_00040_ns_0_b.mtx");
    REQUIRE(ok);

    for (int block_dim : {1, 2, 3})
    {
        auto solver = Solver::create("MAS", "");
        json params;
        params["MAS"]["relative_tolerance"] = 0.0;
        params["MAS"]["absolute_tolerance"] = 1e-6;
        params["MAS"]["use_preconditioned_residual_norm"] = false;
        solver->set_block_size(block_dim);
        solver->set_parameters(params);

        // Eigen::VectorXd b(A.rows());
        // b.setRandom();
        Eigen::VectorXd x(b.size());
        x.setZero();

        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);

        const double err = (A * x - b).norm();
        INFO("block_dim: " + std::to_string(block_dim));
        REQUIRE(err < 1e-6);
    }
}

TEST_CASE("hybrid_convergence", "[.][solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_contact.mtx");
    REQUIRE(ok);
    json solver_info;

    std::vector<std::string> solvers;
#ifdef POLYSOLVE_WITH_CPU_HYBRID
    solvers.push_back("CPUHybrid");
#endif
#ifdef POLYSOLVE_WITH_GPU_HYBRID
    solvers.push_back("GPUHybrid");
#endif

    for (const auto &s : solvers)
    {
        auto solver = Solver::create(s, "");
        json params;
        params[s]["relative_tolerance"] = 0;
        params[s]["absolute_tolerance"] = 1e-12;

        solver->set_block_size(3);
        solver->set_parameters(params);
        Eigen::VectorXd b(A.rows());
        b.setRandom();
        Eigen::VectorXd x(b.size());
        x.setZero();

        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);

        REQUIRE(solver->name() == s);

        solver->get_info(solver_info);

        const double err = (A * x - b).norm();
        INFO("solver: " + s);
        REQUIRE(err < 1e-4);
        REQUIRE(solver_info["num_iterations"] < 500);
    }
}

// Checks set_block_mapping (HYPRE_BoomerAMGSetDofFunc) for Hypre and the CPU/GPU
// hybrid solvers: an explicit mapping that reproduces HYPRE's own default
// interleaved dof_func (row i -> function i % block_size) should converge in the
// same number of iterations as never calling set_block_mapping at all. For
// CPUHybrid in particular this also exercises slicing the (globally-indexed)
// mapping down to each MPI rank's local row range.
TEST_CASE("block_mapping", "[.][solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_contact.mtx");
    REQUIRE(ok);

    const int block_size = 3;
    REQUIRE(A.rows() % block_size == 0);

    Eigen::VectorXi block_mapping(A.rows());
    for (int i = 0; i < A.rows(); ++i)
        block_mapping[i] = i % block_size;

    std::vector<std::string> solvers;
#ifdef POLYSOLVE_WITH_HYPRE
    solvers.push_back("Hypre");
#endif
#ifdef POLYSOLVE_WITH_CPU_HYBRID
    solvers.push_back("CPUHybrid");
#endif
#ifdef POLYSOLVE_WITH_GPU_HYBRID
    solvers.push_back("GPUHybrid");
#endif

    Eigen::VectorXd b(A.rows());
    b.setRandom();

    for (const auto &s : solvers)
    {
        json params;
        if (s == "Hypre")
        {
            params[s]["tolerance"] = 1e-12;
            // Plain AMG needs ~700 iterations on this contact matrix on one rank,
            // and about twice that once the rows are split across ranks.
            params[s]["max_iter"] = 5000;
        }
        else
        {
            params[s]["relative_tolerance"] = 0;
            params[s]["absolute_tolerance"] = 1e-12;
        }

        // Baseline: default interleaved dof_func, set_block_mapping never called.
        auto baseline_solver = Solver::create(s, "");
        baseline_solver->set_block_size(block_size);
        baseline_solver->set_parameters(params);

        Eigen::VectorXd x_baseline(b.size());
        x_baseline.setZero();
        baseline_solver->analyze_pattern(A, A.rows());
        baseline_solver->factorize(A);
        baseline_solver->solve(b, x_baseline);

        json baseline_info;
        baseline_solver->get_info(baseline_info);

        // Explicit mapping reproducing that same interleaved assignment.
        auto mapped_solver = Solver::create(s, "");
        mapped_solver->set_block_size(block_size);
        mapped_solver->set_block_mapping(block_mapping);
        mapped_solver->set_parameters(params);

        Eigen::VectorXd x_mapped(b.size());
        x_mapped.setZero();
        mapped_solver->analyze_pattern(A, A.rows());
        mapped_solver->factorize(A);
        mapped_solver->solve(b, x_mapped);

        json mapped_info;
        mapped_solver->get_info(mapped_info);

        INFO("solver: " + s);
        const double err = (A * x_mapped - b).norm();
        REQUIRE(err < 1e-3);
        // An explicit dof_func matching HYPRE's own default interleaved mapping
        // should make AMG behave identically to never calling set_block_mapping.
        REQUIRE(mapped_info["num_iterations"] == baseline_info["num_iterations"]);
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

TEST_CASE("uncompressed matrices never give a silently wrong solution", "[solver]")
{
    const std::string s = GENERATE("Eigen::SimplicialLDLT", "Eigen::SparseLU", "Eigen::AccelerateLLT", "Eigen::AccelerateLDLT");
    const auto available = Solver::available_solvers();
    if (std::find(available.begin(), available.end(), s) == available.end())
        return; // e.g. the Accelerate solvers only exist on Apple platforms

    // We reserve more room per column than we fill, which is what assembling with insert() or coeffRef() does. Eigen
    // then leaves unused slots between the columns, i.e. the matrix is not in compressed storage.
    const int n = 20;
    StiffnessMatrix A(n, n);
    A.reserve(Eigen::VectorXi::Constant(n, 8));
    for (int i = 0; i < n; ++i)
    {
        A.insert(i, i) = 4;
        if (i + 1 < n)
        {
            A.insert(i + 1, i) = -1;
            A.insert(i, i + 1) = -1;
        }
    }
    REQUIRE(!A.isCompressed());

    const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, 1, 2);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

    auto solver = Solver::create(s, "");
    INFO("solver: " + s);

    // A solver either handles the matrix as it is or refuses it. The Accelerate solvers refuse, because they cannot
    // read this storage.
    bool refused = false;
    try
    {
        solver->analyze_pattern(A, A.rows());
        solver->factorize(A);
        solver->solve(b, x);
    }
    catch (const std::runtime_error &)
    {
        refused = true;
    }
    if (!refused)
        CHECK((A * x - b).norm() < 1e-10);

    // Once compressed, every solver has to solve it.
    A.makeCompressed();
    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    CHECK((A * x - b).norm() < 1e-10);
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

#if defined(POLYSOLVE_WITH_HYPRE) && defined(POLYSOLVE_TEST_RANKS)
// Hypre splits the rows across the nano-mpi ranks and must converge the same
// whatever their number.
TEST_CASE("hypre_ranks", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    Eigen::VectorXd b(A.rows());
    b.setRandom();

    for (const int n : {1, 2, 3, 7})
    {
        ScopedNumRanks ranks(n);
        auto solver = Solver::create("Hypre", "");

        // twice, to check that the solver can be factorized again
        for (int k = 0; k < 2; ++k)
        {
            Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
            solver->analyze_pattern(A, A.rows());
            solver->factorize(A);
            solver->solve(b, x);

            json info;
            solver->get_info(info);
            INFO("ranks: " << n);
            CHECK(info["num_ranks"] == n);
            REQUIRE((A * x - b).norm() < 1e-8);
        }
    }
}

// Ranks with no rows at all.
TEST_CASE("hypre_more_ranks_than_rows", "[solver]")
{
    const int n = 5;
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; ++i)
    {
        triplets.emplace_back(i, i, 2.0);
        if (i > 0)
            triplets.emplace_back(i, i - 1, -1.0);
        if (i + 1 < n)
            triplets.emplace_back(i, i + 1, -1.0);
    }
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());

    ScopedNumRanks ranks(8);
    auto solver = Solver::create("Hypre", "");
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, 1, n);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);

    json info;
    solver->get_info(info);
    CHECK(info["num_ranks"] == 8);
    REQUIRE((A * x - b).norm() < 1e-8);
}

// Only the thread that started the ranks can drive them. A solver used from
// another thread meanwhile solves on that thread alone, and the ranks keep
// working for the thread that owns them.
TEST_CASE("hypre_ranks_other_thread", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    Eigen::VectorXd b(A.rows());
    b.setRandom();

    ScopedNumRanks ranks(3);
    auto on_ranks = Solver::create("Hypre", "");
    on_ranks->analyze_pattern(A, A.rows());
    on_ranks->factorize(A);

    // Catch2 assertions are not thread safe: check the results afterwards.
    json other_info;
    double other_err = -1;
    std::thread([&] {
        auto alone = Solver::create("Hypre", "");
        Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
        alone->analyze_pattern(A, A.rows());
        alone->factorize(A);
        alone->solve(b, x);
        alone->get_info(other_info);
        other_err = (A * x - b).norm();
    }).join();

    CHECK(other_info["num_ranks"] == 1);
    REQUIRE(other_err >= 0);
    REQUIRE(other_err < 1e-8);

    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
    on_ranks->solve(b, x);
    json info;
    on_ranks->get_info(info);
    CHECK(info["num_ranks"] == 3);
    REQUIRE((A * x - b).norm() < 1e-8);
}

#ifdef POLYSOLVE_WITH_CPU_HYBRID
// nano-mpi allows one set of ranks per process, so Hypre and CPUHybrid share
// it; both must work while the other is alive, in any order.
TEST_CASE("hypre_and_cpu_hybrid_share_ranks", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    Eigen::VectorXd b(A.rows());
    b.setRandom();

    ScopedNumRanks ranks(3);
    auto hypre = Solver::create("Hypre", "");
    auto hybrid = Solver::create("CPUHybrid", "");
    json params;
    params["CPUHybrid"]["relative_tolerance"] = 0.0;
    params["CPUHybrid"]["absolute_tolerance"] = 1e-10;
    hybrid->set_parameters(params);

    hypre->analyze_pattern(A, A.rows());
    hybrid->analyze_pattern(A, A.rows());

    for (int k = 0; k < 2; ++k)
    {
        hypre->factorize(A);
        hybrid->factorize(A);

        Eigen::VectorXd x_hypre = Eigen::VectorXd::Zero(b.size());
        Eigen::VectorXd x_hybrid = Eigen::VectorXd::Zero(b.size());
        hybrid->solve(b, x_hybrid);
        hypre->solve(b, x_hypre);

        json info;
        hypre->get_info(info);
        CHECK(info["num_ranks"] == 3);
        REQUIRE((A * x_hypre - b).norm() < 1e-8);
        REQUIRE((A * x_hybrid - b).norm() < 1e-8);
    }

    // The ranks outlive the solver that started them while another uses them.
    hybrid.reset();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
    hypre->factorize(A);
    hypre->solve(b, x);
    REQUIRE((A * x - b).norm() < 1e-8);
}
#endif
#endif

#ifdef POLYSOLVE_WITH_HYPRE
#ifdef POLYSOLVE_WITH_HYPRE_GPU
TEST_CASE("hypre_gpu", "[solver]")
{
    const std::string path = POLYFEM_DATA_DIR;
    Eigen::SparseMatrix<double> A;
    const bool ok = loadMarket(A, path + "/A_2.mat");
    REQUIRE(ok);

    auto solver = Solver::create("Hypre", "");
    json params;
    params["Hypre"]["use_gpu"] = true;
    solver->set_parameters(params);

    Eigen::VectorXd b(A.rows());
    b.setRandom();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);

    json info;
    solver->get_info(info);
    CHECK(info["on_gpu"] == true);
    CHECK(info["num_ranks"] == 1);
    REQUIRE((A * x - b).norm() < 1e-8);

    // and back to the CPU
    params["Hypre"]["use_gpu"] = false;
    solver->set_parameters(params);
    x.setZero();
    solver->factorize(A);
    solver->solve(b, x);
    solver->get_info(info);
    CHECK(info["on_gpu"] == false);
    REQUIRE((A * x - b).norm() < 1e-8);
}
#else
TEST_CASE("hypre_gpu_unavailable", "[solver]")
{
    auto solver = Solver::create("Hypre", "");
    json params;
    params["Hypre"]["use_gpu"] = true;
    REQUIRE_THROWS(solver->set_parameters(params));
}
#endif
#endif

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
        solver->set_block_size(3);
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
        solver->set_block_size(2);
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
        solver->set_block_size(3);
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
        params["AMGCL"]["solver_type"] = "bicgstab";
        solver->set_block_size(3);
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
