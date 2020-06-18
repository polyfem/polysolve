[![Build Status](https://travis-ci.com/polyfem/polysolve.svg?branch=master)](https://travis-ci.com/polyfem/polysolve)
[![Build status](https://ci.appveyor.com/api/projects/status/tk7mfelpp469vqb5/branch/master?svg=true)](https://ci.appveyor.com/project/teseoch/polysolve/branch/master)


# Polysolve

This library contains a cross-platform Eigen Wrappers for many different external solvers including (but not limited to):
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

You can use `LinearSolver::availableSolvers()` to obtain the list of available solvers.


### Parameters for iterative solvers (AMGCL, Eigen Internal Solvers, HYPRE)

 - `max_iter` controls the solver's iterations, default `1000`
 - `conv_tol`, `tolerance` controls the convergence tolerance, default `1e-10`

**Hypre Only**

- `pre_max_iter`, number of pre iterations, default `1`


### Paramters for Pardiso

'mtype', sets the matrix type, default 11
| mtype | Description                             |
|-------|-----------------------------------------|
|    1  | real and structurally symmetric         |
|    2  | real and symmetric positive definite    |
|   -2  | real and symmetric indefinite           |
|    3  | complex and structurally symmetric      |
|    4  | complex and Hermitian positive definite |
|   -4  | complex and Hermitian indefinite        |
|    6  | complex and symmetric                   |
|   11  | real and nonsymmetric                   |
|   13  | complex and nonsymmetric                |
