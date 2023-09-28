#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve::linear
{

    ///
    /// @brief         { Solve a linear system Ax = b with Dirichlet boundary
    ///                conditions. For each Dirichlet node i, we want to ensure that
    ///                x[i] = b[i]. The implementation follows
    ///                http://www.math.colostate.edu/~bangerth/videos.676.21.65.html
    ///
    ///                For memory efficiency, this function updates in place the
    ///                matrix of the linear system. We also return the modified rhs
    ///                to the user }
    ///
    /// @param[in]     solver           { Linear solver class to use for solving the
    ///                                 system }
    /// @param[in,out] A                { Matrix of the linear system without
    ///                                 boundary conditions. Output: modified
    ///                                 matrix. }
    /// @param[in,out] b                { Right-hand side of the linear system.
    ///                                 Output: modified rhs for the equivalent
    ///                                 system. }
    /// @param[in]     dirichlet_nodes  { List of ids of Dirichlet nodes }
    /// @param[in,out] x                { Unknown vector }
    ///
    Eigen::Vector4d dirichlet_solve(Solver &solver, StiffnessMatrix &A,
                                    Eigen::VectorXd &b, const std::vector<int> &dirichlet_nodes, Eigen::VectorXd &x,
                                    const int precond_num,
                                    const std::string &save_path = "", bool compute_spectrum = false,
                                    const bool remove_zero_cols = false,
                                    const bool skip_last_cols = false);

    void prefactorize(Solver &solver, StiffnessMatrix &A,
                      const std::vector<int> &dirichlet_nodes, const int precond_num,
                      const std::string &save_path = "");

    void dirichlet_solve_prefactorized(Solver &solver, const StiffnessMatrix &A, Eigen::VectorXd &f,
                                       const std::vector<int> &dirichlet_nodes, Eigen::VectorXd &u);

} // namespace polysolve::linear
