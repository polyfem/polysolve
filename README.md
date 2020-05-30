[![Build Status](https://travis-ci.com/polyfem/solver-warpper.svg?branch=master)](https://travis-ci.com/polyfem/solver-warpper)
[![Build status](https://ci.appveyor.com/api/projects/status/2ry9mbgd14mb9hlf/branch/master?svg=true)](https://ci.appveyor.com/project/teseoch/solver-warpper/branch/master)


# PolyFEM Solvers Wrappers

This library containts an Eigen Wrappers for many different external solvers including (but not limited to):
 - Hypre
 - AMGCL
 - Pardiso


## Example Usage

```c++
const std::string solver_name = "Hypre"
auto solver = LinearSolver::create(solver_name, "");

// configuration parameters like iteration or accuracy for iterative solvers
// solver->setParameters(params);

Eigen::VectorXd b;
Eigen::VectorXd x(b.size());

solver->analyzePattern(A, A.rows());
solver->factorize(A);
solver->solve(b, x);
```