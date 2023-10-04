////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"
#include "EigenSolver.hpp"
#include "SaddlePointSolver.hpp"

// -----------------------------------------------------------------------------
#include <Eigen/Sparse>
#ifdef POLYSOLVE_WITH_ACCELERATE
#include <Eigen/AccelerateSupport>
#endif
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
#include "Pardiso.hpp"
#endif
#ifdef POLYSOLVE_WITH_HYPRE
#include "HypreSolver.hpp"
#endif
#ifdef POLYSOLVE_WITH_AMGCL
#include "AMGCL.hpp"
#endif
#ifdef POLYSOLVE_WITH_CUSOLVER
#include "CuSolverDN.cuh"
#endif
#include <unsupported/Eigen/IterativeSolvers>

////////////////////////////////////////////////////////////////////////////////

namespace polysolve::linear
{
    using polysolve::StiffnessMatrix;

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

#define RETURN_DIRECT_SOLVER_PTR(EigenSolver, Name)      \
    do                                                   \
    {                                                    \
        return std::make_unique<EigenDirect<EigenSolver< \
            polysolve::linear::StiffnessMatrix>>>(Name); \
    } while (0)

#define RETURN_DIRECT_DENSE_SOLVER_PTR(EigenSolver, Name)     \
    do                                                        \
    {                                                         \
        return std::make_unique<EigenDenseSolver<EigenSolver< \
            Eigen::MatrixXd>>>(Name);                         \
    } while (0)

    ////////////////////////////////////////////////////////////////////////////////

    namespace
    {

        template <template <class, class> class SparseSolver, typename Precond>
        struct MakeSolver
        {
            typedef EigenIterative<SparseSolver<StiffnessMatrix, Precond>> type;
        };

        template <template <class, int, class> class SparseSolver, typename Precond>
        struct MakeSolverSym
        {
            typedef EigenIterative<SparseSolver<StiffnessMatrix,
                                                Eigen::Lower | Eigen::Upper, Precond>>
                type;
        };

        // -----------------------------------------------------------------------------

        template <
            template <class, class> class SolverType,
            typename DefaultPrecond = Eigen::DiagonalPreconditioner<double>>
        struct PrecondHelper
        {
            static std::unique_ptr<Solver> create(const std::string &arg, const std::string &name)
            {
                ENUMERATE_PRECOND(MakeSolver, SolverType, DefaultPrecond, arg, name);
            }
        };

        template <
            template <class, int, class> class SolverType,
            typename DefaultPrecond = Eigen::DiagonalPreconditioner<double>>
        struct PrecondHelperSym
        {
            static std::unique_ptr<Solver> create(const std::string &arg, const std::string &name)
            {
                ENUMERATE_PRECOND(MakeSolverSym, SolverType, DefaultPrecond, arg, name);
            }
        };

    } // anonymous namespace

    ////////////////////////////////////////////////////////////////////////////////

    // Static constructor
    std::unique_ptr<Solver> Solver::create(const std::string &solver, const std::string &precond)
    {
        using namespace Eigen;

        if (solver.empty() || solver == "Eigen::SimplicialLDLT")
        {
            RETURN_DIRECT_SOLVER_PTR(SimplicialLDLT, "Eigen::SimplicialLDLT");
        }
        else if (solver == "Eigen::SparseLU")
        {
            RETURN_DIRECT_SOLVER_PTR(SparseLU, "Eigen::SparseLU");
#ifdef POLYSOLVE_WITH_ACCELERATE
        }
        else if (solver == "Eigen::AccelerateLLT")
        {
            RETURN_DIRECT_SOLVER_PTR(AccelerateLLT, "Eigen::AccelerateLLT");
        }
        else if (solver == "Eigen::AccelerateLDLT")
        {
            RETURN_DIRECT_SOLVER_PTR(AccelerateLDLT, "Eigen::AccelerateLDLT");
#endif
#ifdef POLYSOLVE_WITH_CHOLMOD
        }
        else if (solver == "Eigen::CholmodSupernodalLLT")
        {
            RETURN_DIRECT_SOLVER_PTR(CholmodSupernodalLLT, "Eigen::CholmodSupernodalLLT");
        }
        else if (solver == "Eigen::CholmodDecomposition")
        {
            RETURN_DIRECT_SOLVER_PTR(CholmodDecomposition, "Eigen::CholmodDecomposition");
        }
        else if (solver == "Eigen::CholmodSimplicialLLT")
        {
            RETURN_DIRECT_SOLVER_PTR(CholmodSimplicialLLT, "Eigen::CholmodSimplicialLLT");
        }
        else if (solver == "Eigen::CholmodSimplicialLDLT")
        {
            RETURN_DIRECT_SOLVER_PTR(CholmodSimplicialLDLT, "Eigen::CholmodSimplicialLDLT");
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
            return std::make_unique<Pardiso>();
#endif
#ifdef POLYSOLVE_WITH_CUSOLVER
        }
        else if (solver == "cuSolverDN")
        {
            return std::make_unique<CuSolverDN<double>>();
        }
        else if (solver == "cuSolverDN_float")
        {
            return std::make_unique<CuSolverDN<float>>();
#endif
#ifdef POLYSOLVE_WITH_HYPRE
        }
        else if (solver == "Hypre")
        {
            return std::make_unique<HypreSolver>();
#endif
#ifdef POLYSOLVE_WITH_AMGCL
        }
        else if (solver == "AMGCL")
        {
            return std::make_unique<AMGCL>();
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
        /////DENSE Eigen
        else if (solver == "Eigen::PartialPivLU")
        {
            RETURN_DIRECT_DENSE_SOLVER_PTR(PartialPivLU, "Eigen::PartialPivLU");
        }
        else if (solver == "Eigen::FullPivLU")
        {
            RETURN_DIRECT_DENSE_SOLVER_PTR(FullPivLU, "Eigen::FullPivLU");
        }
        else if (solver == "Eigen::HouseholderQR")
        {
            RETURN_DIRECT_DENSE_SOLVER_PTR(HouseholderQR, "Eigen::HouseholderQR");
        }
        else if (solver == "Eigen::ColPivHouseholderQR")
        {
            RETURN_DIRECT_DENSE_SOLVER_PTR(ColPivHouseholderQR, "Eigen::ColPivHouseholderQR");
        }
        else if (solver == "Eigen::FullPivHouseholderQR")
        {
            RETURN_DIRECT_DENSE_SOLVER_PTR(FullPivHouseholderQR, "Eigen::FullPivHouseholderQR");
        }
        else if (solver == "Eigen::CompleteOrthogonalDecomposition")
        {
            RETURN_DIRECT_DENSE_SOLVER_PTR(CompleteOrthogonalDecomposition, "Eigen::CompleteOrthogonalDecomposition");
        }
        else if (solver == "Eigen::LLT")
        {
            RETURN_DIRECT_DENSE_SOLVER_PTR(LLT, "Eigen::LLT");
        }
        else if (solver == "Eigen::LDLT")
        {
            RETURN_DIRECT_DENSE_SOLVER_PTR(LDLT, "Eigen::LDLT");
        }
        // else if (solver == "Eigen::BDCSVD")
        // {
        //     RETURN_DIRECT_DENSE_SOLVER_PTR(BDCSVD, "Eigen::BDCSVD");
        // }
        // else if (solver == "Eigen::JacobiSVD")
        // {
        //     RETURN_DIRECT_DENSE_SOLVER_PTR(JacobiSVD, "Eigen::JacobiSVD");
        // }
        throw std::runtime_error("Unrecognized solver type: " + solver);
    }

    ////////////////////////////////////////////////////////////////////////////////

    // List available solvers
    std::vector<std::string> Solver::available_solvers()
    {
        return {{
            "Eigen::SimplicialLDLT",
            "Eigen::SparseLU",
#ifdef POLYSOLVE_WITH_ACCELERATE
            "Eigen::AccelerateLLT",
            "Eigen::AccelerateLDLT",
#endif
#ifdef POLYSOLVE_WITH_CHOLMOD
            "Eigen::CholmodSupernodalLLT",
            "Eigen::CholmodDecomposition",
            "Eigen::CholmodSimplicialLLT",
            "Eigen::CholmodSimplicialLDLT",
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
            "cuSolverDN_float",
#endif
#ifdef POLYSOLVE_WITH_HYPRE
            "Hypre",
#endif
#ifdef POLYSOLVE_WITH_AMGCL
            "AMGCL",
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
            "Eigen::PartialPivLU",
            "Eigen::FullPivLU",
            "Eigen::HouseholderQR",
            "Eigen::ColPivHouseholderQR",
            "Eigen::FullPivHouseholderQR",
            "Eigen::CompleteOrthogonalDecomposition",
            "Eigen::LLT",
            "Eigen::LDLT"
            // "Eigen::BDCSVD",
            // "Eigen::JacobiSVD"
        }};
    }

    std::string Solver::defaultSolver()
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
    std::vector<std::string> Solver::availablePrecond()
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

    std::string Solver::defaultPrecond()
    {
        return "Eigen::DiagonalPreconditioner";
    }

    ////////////////////////////////////////////////////////////////////////////////

} // namespace polysolve::linear
