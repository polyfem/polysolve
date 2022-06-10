![Build](https://github.com/polyfem/polysolve/workflows/Build/badge.svg)



# PolySolve

This library contains a cross-platform Eigen wrapper for many different external linear solvers including (but not limited to):

 - CHOLMOD
 - Hypre
 - AMGCL
 - Pardiso


## Example Usage

```c++
const std::string solver_name = "Hypre"
auto solver = LinearSolver::create(solver_name, "");

// Configuration parameters like iteration or accuracy for iterative solvers
// solver->setParameters(params);

// System sparse matrix
Eigen::SparseMatrix<double> A;

// Right-hand side
Eigen::VectorXd b;

// Solution
Eigen::VectorXd x(b.size());

solver->analyzePattern(A, A.rows());
solver->factorize(A);
solver->solve(b, x);
```

You can use `LinearSolver::availableSolvers()` to obtain the list of available solvers.

## Parameters

Polysolve uses a json file to provide parameters to the individual solvers. The following template can be used as a starting points, and a more detailed explanation of the parameters is below.

```json
{
    "Eigen::LeastSquaresConjugateGradient": {
        "max_iter": 1000,
        "tolerance": 1e-6
    },
    "Eigen::DGMRES": {
        "max_iter": 1000,
        "tolerance": 1e-6
    },
    "Eigen::ConjugateGradient": {
        "max_iter": 1000,
        "tolerance": 1e-6
    },
    "Eigen::BiCGSTAB": {
        "max_iter": 1000,
        "tolerance": 1e-6
    },
    "Eigen::GMRES": {
        "max_iter": 1000,
        "tolerance": 1e-6
    },
    "Eigen::MINRES": {
        "max_iter": 1000,
        "tolerance": 1e-6
    },
    "Pardiso": {
        "mtype": -1
    },
    "Hypre": {
        "max_iter": 1000,
        "pre_max_iter": 1000,
        "tolerance": 1e-6
    },
    "AMGCL": {
        "precond": {
            "relax": {
                "degree": 16,
                "type": "chebyshev",
                "power_iters": 100,
                "higher": 2,
                "lower": 0.008333333333,
                "scale": true
            },
            "class": "amg",
            "max_levels": 6,
            "direct_coarse": false,
            "ncycle": 2,
            "coarsening": {
                "type": "smoothed_aggregation",
                "estimate_spectral_radius": true,
                "relax": 1,
                "aggr": {
                    "eps_strong": 0
                }
            }
        },
        "solver": {
            "tol": 1e-10,
            "maxiter": 1000,
            "type": "cg"
        }
    }
}
```

###  Iterative solvers (AMGCL, Eigen Internal Solvers, HYPRE)

 - `max_iter` controls the solver's iterations, default `1000`
 - `conv_tol`, `tolerance` controls the convergence tolerance, default `1e-10`

#### Hypre Only

- `pre_max_iter`, number of pre iterations, default `1`

#### AMGCL Only

The default parameters of the AMGCL solver are:
```json
{
    "precond": {
        "relax": {
            "degree": 16,
            "type": "chebyshev",
            "power_iters": 100,
            "higher": 2,
            "lower": 0.008333333333,
            "scale": true
        },
        "class": "amg",
        "max_levels": 6,
        "direct_coarse": false,
        "ncycle": 2,
        "coarsening": {
            "type": "smoothed_aggregation",
            "estimate_spectral_radius": true,
            "relax": 1,
            "aggr": {
                "eps_strong": 0
            }
        }
    },
    "solver": {
        "tol": 1e-10,
        "maxiter": 1000,
        "type": "cg"
    }
}
```

For a more details and options refer to the [AMGCL documentation](https://amgcl.readthedocs.io/en/latest/components.html).

### Pardiso

`mtype`, sets the matrix type, default 11

| mtype | Description                             |
| ----- | --------------------------------------- |
| 1     | real and structurally symmetric         |
| 2     | real and symmetric positive definite    |
| -2    | real and symmetric indefinite           |
| 3     | complex and structurally symmetric      |
| 4     | complex and Hermitian positive definite |
| -4    | complex and Hermitian indefinite        |
| 6     | complex and symmetric                   |
| 11    | real and nonsymmetric                   |
| 13    | complex and nonsymmetric                |
