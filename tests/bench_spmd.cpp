// Time polysolve's hybrid solver as an SPMD program: every rank enters the
// solve, which is how the solver is meant to be driven. The same source runs
// under mpirun -np N (ranks are processes) and against nano-mpi (ranks are
// threads of this process), so the two backends are compared on one driver.
#include <polysolve/linear/Solver.hpp>
#include <Eigen/Sparse>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv)
{
    const int n = (argc > 1) ? std::atoi(argv[1]) : 60;   // grid edge
    const long N = (long) n * n * n;

    std::vector<Eigen::Triplet<double>> T;
    T.reserve(7 * (size_t) N);
    auto id = [n](int i, int j, int k) { return (long)(i * n + j) * n + k; };
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
            {
                const long r = id(i, j, k);
                T.emplace_back(r, r, 6.0);
                if (i > 0)     T.emplace_back(r, id(i - 1, j, k), -1.0);
                if (i < n - 1) T.emplace_back(r, id(i + 1, j, k), -1.0);
                if (j > 0)     T.emplace_back(r, id(i, j - 1, k), -1.0);
                if (j < n - 1) T.emplace_back(r, id(i, j + 1, k), -1.0);
                if (k > 0)     T.emplace_back(r, id(i, j, k - 1), -1.0);
                if (k < n - 1) T.emplace_back(r, id(i, j, k + 1), -1.0);
            }
    Eigen::SparseMatrix<double> A((int) N, (int) N);
    A.setFromTriplets(T.begin(), T.end());
    T.clear(); T.shrink_to_fit();

    Eigen::VectorXd b = Eigen::VectorXd::Ones((int) N);
    Eigen::VectorXd x = Eigen::VectorXd::Zero((int) N);

    const std::string name = (argc > 2) ? argv[2] : "CPUHybrid";
    using clk = std::chrono::high_resolution_clock;

    auto solver = polysolve::linear::Solver::create(name, "");
    const auto t0 = clk::now();
    solver->analyze_pattern(A, (int) N);
    solver->factorize(A);
    const auto t1 = clk::now();
    solver->solve(b, x);
    const auto t2 = clk::now();

    const double setup = std::chrono::duration<double>(t1 - t0).count();
    const double solve = std::chrono::duration<double>(t2 - t1).count();
    const double res   = (A * x - b).norm() / b.norm();
    std::printf("n=%d N=%ld setup=%.4f solve=%.4f total=%.4f relres=%.3e\n",
                n, N, setup, solve, setup + solve, res);
    return 0;
}
