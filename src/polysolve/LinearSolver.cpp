////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolver.hpp>
#include <polysolve/LinearSolverEigen.hpp>
#include <polysolve/SaddlePointSolver.hpp>

// -----------------------------------------------------------------------------
#include <Eigen/Sparse>
#ifdef POLYSOLVE_WITH_CHOLMOD
#include <Eigen/CholmodSupport>
#endif
#ifdef POLYSOLVE_WITH_UMFPACK
#include <Eigen/UmfPackSupport>
#endif
#ifdef POLYSOLVE_WITH_SUPERLU
#include <Eigen/SuperLUSupport>
#endif
#ifdef POLYSOLVE_WITH_MKL
#include <Eigen/PardisoSupport>
#endif
#ifdef POLYSOLVE_WITH_PARDISO
#include <polysolve/LinearSolverPardiso.hpp>
#endif
#ifdef POLYSOLVE_WITH_HYPRE
#include <polysolve/LinearSolverHypre.hpp>
#endif
#ifdef POLYSOLVE_WITH_AMGCL
#include <polysolve/LinearSolverAMGCL.hpp>
#ifdef POLYSOLVE_WITH_CUDA
#include <polysolve/LinearSolverAMGCL_cuda.hpp>
#endif
#endif
#ifdef POLYSOLVE_WITH_CUSOLVER
#include <polysolve/LinearSolverCuSolverDN.cuh>
#endif
#ifdef POLYSOLVE_WITH_PETSC
#include <polysolve/LinearSolverPETSC.hpp>
#endif
#include <unsupported/Eigen/IterativeSolvers>

////////////////////////////////////////////////////////////////////////////////

namespace polysolve
{

    ////////////////////////////////////////////////////////////////////////////////

#if EIGEN_VERSION_AT_LEAST(3, 3, 0)

// Magic macro because C++ has no introspection
#define ENUMERATE_PRECOND(HelperFunctor, SolverType, DefaultPrecond, precond, name)                                 \
    do                                                                                                              \
    {                                                                                                               \
        using namespace Eigen;                                                                                      \
        if (precond == "Eigen::IdentityPreconditioner")                                                             \
        {                                                                                                           \
            return std::make_unique<typename HelperFunctor<SolverType,                                              \
                                                           IdentityPreconditioner>::type>(name);                    \
        }                                                                                                           \
        else if (precond == "Eigen::DiagonalPreconditioner")                                                        \
        {                                                                                                           \
            return std::make_unique<typename HelperFunctor<SolverType,                                              \
                                                           DiagonalPreconditioner<double>>::type>(name);            \
        }                                                                                                           \
        else if (precond == "Eigen::IncompleteCholesky")                                                            \
        {                                                                                                           \
            return std::make_unique<typename HelperFunctor<SolverType,                                              \
                                                           IncompleteCholesky<double>>::type>(name);                \
        }                                                                                                           \
        else if (precond == "Eigen::LeastSquareDiagonalPreconditioner")                                             \
        {                                                                                                           \
            return std::make_unique<typename HelperFunctor<SolverType,                                              \
                                                           LeastSquareDiagonalPreconditioner<double>>::type>(name); \
        }                                                                                                           \
        else if (precond == "Eigen::IncompleteLUT")                                                                 \
        {                                                                                                           \
            return std::make_unique<typename HelperFunctor<SolverType,                                              \
                                                           IncompleteLUT<double>>::type>(name);                     \
        }                                                                                                           \
        else                                                                                                        \
        {                                                                                                           \
            return std::make_unique<typename HelperFunctor<SolverType,                                              \
                                                           DefaultPrecond>::type>(name);                            \
        }                                                                                                           \
    } while (0)

#else

// Magic macro because C++ has no introspection
#define ENUMERATE_PRECOND(HelperFunctor, SolverType, DefaultPrecond, precond)                        \
    do                                                                                               \
    {                                                                                                \
        using namespace Eigen;                                                                       \
        if (precond == "Eigen::IdentityPreconditioner")                                              \
        {                                                                                            \
            return std::make_unique<typename HelperFunctor<SolverType,                               \
                                                           IdentityPreconditioner>::type>();         \
        }                                                                                            \
        else if (precond == "Eigen::DiagonalPreconditioner")                                         \
        {                                                                                            \
            return std::make_unique<typename HelperFunctor<SolverType,                               \
                                                           DiagonalPreconditioner<double>>::type>(); \
        }                                                                                            \
        else if (precond == "Eigen::IncompleteCholesky")                                             \
        {                                                                                            \
            return std::make_unique<typename HelperFunctor<SolverType,                               \
                                                           IncompleteCholesky<double>>::type>();     \
        }                                                                                            \
        else if (precond == "Eigen::IncompleteLUT")                                                  \
        {                                                                                            \
            return std::make_unique<typename HelperFunctor<SolverType,                               \
                                                           IncompleteLUT<double>>::type>();          \
        }                                                                                            \
        else                                                                                         \
        {                                                                                            \
            return std::make_unique<typename HelperFunctor<SolverType,                               \
                                                           DefaultPrecond>::type>();                 \
        }                                                                                            \
    } while (0)

#endif

    // -----------------------------------------------------------------------------

#define RETURN_DIRECT_SOLVER_PTR(EigenSolver, Name)                  \
    do                                                               \
    {                                                                \
        return std::make_unique<LinearSolverEigenDirect<EigenSolver< \
            polysolve::StiffnessMatrix>>>(Name);                     \
    } while (0)

    ////////////////////////////////////////////////////////////////////////////////

    namespace
    {

        template <template <class, class> class SparseSolver, typename Precond>
        struct MakeSolver
        {
            typedef LinearSolverEigenIterative<SparseSolver<StiffnessMatrix, Precond>> type;
        };

        template <template <class, int, class> class SparseSolver, typename Precond>
        struct MakeSolverSym
        {
            typedef LinearSolverEigenIterative<SparseSolver<StiffnessMatrix,
                                                            Eigen::Lower | Eigen::Upper, Precond>>
                type;
        };

        // -----------------------------------------------------------------------------

        template <
            template <class, class> class SolverType,
            typename DefaultPrecond = Eigen::DiagonalPreconditioner<double>>
        struct PrecondHelper
        {
            static std::unique_ptr<LinearSolver> create(const std::string &arg, const std::string &name)
            {
                ENUMERATE_PRECOND(MakeSolver, SolverType, DefaultPrecond, arg, name);
            }
        };

        template <
            template <class, int, class> class SolverType,
            typename DefaultPrecond = Eigen::DiagonalPreconditioner<double>>
        struct PrecondHelperSym
        {
            static std::unique_ptr<LinearSolver> create(const std::string &arg, const std::string &name)
            {
                ENUMERATE_PRECOND(MakeSolverSym, SolverType, DefaultPrecond, arg, name);
            }
        };

    } // anonymous namespace

    ////////////////////////////////////////////////////////////////////////////////

    // Static constructor
    std::unique_ptr<LinearSolver> LinearSolver::create(const std::string &solver, const std::string &precond)
    {
        using namespace Eigen;

        if (solver.empty() || solver == "Eigen::SimplicialLDLT")
        {
            RETURN_DIRECT_SOLVER_PTR(SimplicialLDLT, "Eigen::SimplicialLDLT");
        }
        else if (solver == "Eigen::SparseLU")
        {
            RETURN_DIRECT_SOLVER_PTR(SparseLU, "Eigen::SparseLU");
#ifdef POLYSOLVE_WITH_CHOLMOD
        }
        else if (solver == "Eigen::CholmodSupernodalLLT")
        {
            RETURN_DIRECT_SOLVER_PTR(CholmodSupernodalLLT, "Eigen::CholmodSupernodalLLT");
#endif
#ifdef POLYSOLVE_WITH_UMFPACK
#ifndef POLYSOLVE_LARGE_INDEX
        }
        else if (solver == "Eigen::UmfPackLU")
        {
            RETURN_DIRECT_SOLVER_PTR(UmfPackLU, "Eigen::UmfPackLU");
#endif
#endif
#ifdef POLYSOLVE_WITH_SUPERLU
        }
        else if (solver == "Eigen::SuperLU")
        {
            RETURN_DIRECT_SOLVER_PTR(SuperLU, "Eigen::SuperLU");
#endif
#ifdef POLYSOLVE_WITH_MKL
        }
        else if (solver == "Eigen::PardisoLLT")
        {
            RETURN_DIRECT_SOLVER_PTR(PardisoLLT, "Eigen::PardisoLLT");
        }
        else if (solver == "Eigen::PardisoLDLT")
        {
            RETURN_DIRECT_SOLVER_PTR(PardisoLDLT, "Eigen::PardisoLDLT");
        }
        else if (solver == "Eigen::PardisoLU")
        {
            RETURN_DIRECT_SOLVER_PTR(PardisoLU, "Eigen::PardisoLU");
#endif
#ifdef POLYSOLVE_WITH_PARDISO
        }
        else if (solver == "Pardiso")
        {
            return std::make_unique<LinearSolverPardiso>();
#endif
#ifdef POLYSOLVE_WITH_CUSOLVER
        }
        else if (solver == "cuSolverDN")
        {
            return std::make_unique<LinearSolverCuSolverDN>();
#endif
#ifdef POLYSOLVE_WITH_PETSC
        }
        else if (solver == "PETSC_Solver")
        {
            return std::make_unique<LinearSolverPETSC>();
#endif
#ifdef POLYSOLVE_WITH_HYPRE
        }
        else if (solver == "Hypre")
        {
            return std::make_unique<LinearSolverHypre>();
#endif
#ifdef POLYSOLVE_WITH_AMGCL
        }
        else if (solver == "AMGCL")
        {
            return std::make_unique<LinearSolverAMGCL>();
#ifdef POLYSOLVE_WITH_CUDA
        }
        else if (solver == "AMGCL_cuda")
        {
            return std::make_unique<LinearSolverAMGCL_cuda>();
#endif
#endif

#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
            // Available only with Eigen 3.3.0 and newer
#ifndef POLYSOLVE_LARGE_INDEX
        }
        else if (solver == "Eigen::LeastSquaresConjugateGradient")
        {
            return PrecondHelper<BiCGSTAB, LeastSquareDiagonalPreconditioner<double>>::create(precond, "Eigen::LeastSquaresConjugateGradient");
        }
        else if (solver == "Eigen::DGMRES")
        {
            return PrecondHelper<DGMRES>::create(precond, "Eigen::DGMRES");
#endif
#endif
#ifndef POLYSOLVE_LARGE_INDEX
        }
        else if (solver == "Eigen::ConjugateGradient")
        {
            return PrecondHelperSym<ConjugateGradient>::create(precond, "Eigen::ConjugateGradient");
        }
        else if (solver == "Eigen::BiCGSTAB")
        {
            return PrecondHelper<BiCGSTAB>::create(precond, "Eigen::BiCGSTAB");
        }
        else if (solver == "Eigen::GMRES")
        {
            return PrecondHelper<GMRES>::create(precond, "Eigen::GMRES");
        }
        else if (solver == "Eigen::MINRES")
        {
            return PrecondHelperSym<MINRES>::create(precond, "Eigen::MINRES");
#endif
        }
        else if (solver == "SaddlePointSolver")
        {
            return std::make_unique<SaddlePointSolver>();
        }
        throw std::runtime_error("Unrecognized solver type: " + solver);
    }

    ////////////////////////////////////////////////////////////////////////////////

    // List available solvers
    std::vector<std::string> LinearSolver::availableSolvers()
    {
        return {{
            "Eigen::SimplicialLDLT",
            "Eigen::SparseLU",
#ifdef POLYSOLVE_WITH_CHOLMOD
            "Eigen::CholmodSupernodalLLT",
#endif
#ifdef POLYSOLVE_WITH_UMFPACK
            "Eigen::UmfPackLU",
#endif
#ifdef POLYSOLVE_WITH_SUPERLU
            "Eigen::SuperLU",
#endif
#ifdef POLYSOLVE_WITH_MKL
            "Eigen::PardisoLLT",
            "Eigen::PardisoLDLT",
            "Eigen::PardisoLU",
#endif
#ifdef POLYSOLVE_WITH_PARDISO
            "Pardiso",
#endif
#ifdef POLYSOLVE_WITH_CUSOLVER
            "cuSolverDN",
#endif
#ifdef POLYSOLVE_WITH_PETSC
            "PETSC_Solver",
#endif
#ifdef POLYSOLVE_WITH_HYPRE
            "Hypre",
#endif
#ifdef POLYSOLVE_WITH_AMGCL
            "AMGCL",
#ifdef POLYSOLVE_WITH_CUDA
            "AMGCL_cuda",
#endif
#endif
#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
#ifndef POLYSOLVE_LARGE_INDEX
            "Eigen::LeastSquaresConjugateGradient",
            "Eigen::DGMRES",
#endif
#endif
            "Eigen::ConjugateGradient",
            "Eigen::BiCGSTAB",
            "Eigen::GMRES",
            "Eigen::MINRES",
        }};
    }

    std::string LinearSolver::defaultSolver()
    {
        // return "Eigen::BiCGSTAB";
#ifdef POLYSOLVE_WITH_PARDISO
        return "Pardiso";
#else
#ifdef POLYSOLVE_WITH_HYPRE
        return "Hypre";
#else
        return "Eigen::BiCGSTAB";
#endif
#endif
    }

    // -----------------------------------------------------------------------------

    // List available preconditioners
    std::vector<std::string> LinearSolver::availablePrecond()
    {
        return {{
            "Eigen::IdentityPreconditioner",
            "Eigen::DiagonalPreconditioner",
            "Eigen::IncompleteCholesky",
#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
            "Eigen::LeastSquareDiagonalPreconditioner",
#endif
#ifndef POLYSOLVE_LARGE_INDEX
            "Eigen::IncompleteLUT",
#endif
        }};
    }

    std::string LinearSolver::defaultPrecond()
    {
        return "Eigen::DiagonalPreconditioner";
    }

    ////////////////////////////////////////////////////////////////////////////////

} // namespace polysolve
