![Build](https://github.com/polyfem/polysolve/workflows/Build/badge.svg)


# PolySolve

This library contains a cross-platform Eigen wrapper for many different external linear solvers including (but not limited to):

 - Hypre
 - AMGCL
 - Pardiso


## Example Usage

```c++
const std::string solver_name = "Hypre"
auto solver = LinearSolver::create(solver_name, "");

// configuration parameters like iteration or accuracy for iterative solvers
// solver->setParameters(params);

//System sparse matrix
Eigen::SparseMatrix<double> A;

//right-hand side
Eigen::VectorXd b;

//solution
Eigen::VectorXd x(b.size());

solver->analyzePattern(A, A.rows());
solver->factorize(A);
solver->solve(b, x);
```

You can use `LinearSolver::availableSolvers()` to obtain the list of available solvers.

## Parameters

###  Iterative solvers (AMGCL, Eigen Internal Solvers, HYPRE)

 - `max_iter` controls the solver's iterations, default `1000`
 - `conv_tol`, `tolerance` controls the convergence tolerance, default `1e-10`

**Hypre Only**

- `pre_max_iter`, number of pre iterations, default `1`


### Pardiso

`mtype`, sets the matrix type, default 11

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