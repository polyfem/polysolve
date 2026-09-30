# PolySolve

[![Build](https://github.com/polyfem/polysolve/workflows/Build/badge.svg)](https://github.com/polyfem/polysolve/actions/workflows/continuous.yml)
[![codecov](https://codecov.io/github/polyfem/polysolve/graph/badge.svg?token=9CTTZX9A2D)](https://codecov.io/github/polyfem/polysolve)


[![PyPI](https://img.shields.io/pypi/v/polyfem-polysolve.svg)](https://pypi.org/project/polyfem-polysolve/)
[![Python versions](https://img.shields.io/pypi/pyversions/polyfem-polysolve.svg)](https://pypi.org/project/polyfem-polysolve/)
[![CI](https://github.com/polyfem/polysolve-python/actions/workflows/continuous.yml/badge.svg)](https://github.com/polyfem/polysolve-python/actions/workflows/continuous.yml)
[![Build & publish](https://github.com/polyfem/polysolve-python/actions/workflows/release.yml/badge.svg)](https://github.com/polyfem/polysolve-python/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


This library contains:
   - a cross-platform wrapper for many different external linear solvers including (but not limited to):

        - CHOLMOD
        - Hypre
        - AMGCL
        - Pardiso

   - robust non-linear solver
   - Python Bindings


## Example C++ Usage

```c++
const std::string solver_name = "Hypre"
auto solver = Solver::create(solver_name, "");

// Configuration parameters like iteration or accuracy for iterative solvers
// solver->set_parameters(params);

// System sparse matrix
Eigen::SparseMatrix<double> A;

// Right-hand side
Eigen::VectorXd b;

// Solution
Eigen::VectorXd x(b.size());

solver->analyze_pattern(A, A.rows());
solver->factorize(A);
solver->solve(b, x);
```

You can use `Solver::available_solvers()` to obtain the list of available solvers.


## Example Python Usage

Install
```bash
pip install polyfem-polysolve
```

### Non-linear solver

```python
import numpy as np
import scipy.sparse
import polysolve


class Quadratic(polysolve.Problem):
    def value(self, x):
        y = x - np.array([-2.0, 3.0, 1.0])
        return float(y @ y)

    def gradient(self, x):
        return 2.0 * (x - np.array([-2.0, 3.0, 1.0]))

    def hessian(self, x):
        return 2.0 * scipy.sparse.eye(x.size, format="csc")


x, log = polysolve.minimize(
    Quadratic(),
    np.zeros(3),
    {
        "solver": "Newton",
        "line_search": {"method": "Backtracking"},
        "max_iterations": 100,
    },
    {"solver": "Eigen::SimplicialLDLT"},
)

print(x)
print(log)
```

Python subclasses must implement `value(x)`, `gradient(x)`, and `hessian(x)`. Optional PolySolve callbacks such as `solution_changed`,  `stop`, `is_step_valid`, and `max_step_size` can also be implemented on the subclass.

### Linear solves

For a one-off linear system:

```python
A = scipy.sparse.csc_matrix([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])

x = polysolve.solve(A, b, {"solver": "Eigen::SimplicialLDLT"})
```

For repeated solves with the same matrix pattern:

```python
solver = polysolve.LinearSolver({"solver": "Eigen::SimplicialLDLT"})
solver.analyze_pattern(A)
solver.factorise(A)  # factorize(A) is also available

x = solver.solve(b)
print(solver.info())
```


# Parameters

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

## Troubleshooting

### Compilation error: `use of undeclared identifier 'SuiteSparse_config'`

This error is cause by having a more recent version of SuiteSparse (`≥ v7.0.0`) installed on your system than the version we download and build. We use [@sergiud's fork of SuiteSparse](https://github.com/sergiud/SuiteSparse) which includes CMake support. However, the fork is not up to date with the latest version of SuiteSparse (currently `v5.12.0` while the [official release](https://github.com/DrTimothyAldenDavis/SuiteSparse) is at version `v7.0.1`). Version `v7.0.0` changed the `SuiteSparse_config.h` header and no longer includes the necessary struct definitions.

#### Solution

For now, if you can, please downgrade (`< v7.0.0`) or uninstall your system version of SuiteSparse. In the meantime, we will work with the SuiteSparse developers to resolve this issue.
