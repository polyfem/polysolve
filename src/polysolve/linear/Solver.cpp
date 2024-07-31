////////////////////////////////////////////////////////////////////////////////
#include <polysolve/Utils.hpp>

#include "Solver.hpp"
#include "EigenSolver.hpp"
#include "SaddlePointSolver.hpp"

#include <jse/jse.h>
#include <spdlog/spdlog.h>

#include <fstream>

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
#ifdef POLYSOLVE_WITH_TRILINOS
#include "TrilinosSolver.hpp"
#endif
#ifdef POLYSOLVE_WITH_CUSOLVER
#include "CuSolverDN.cuh"
#endif
#include <unsupported/Eigen/IterativeSolvers>

////////////////////////////////////////////////////////////////////////////////

namespace polysolve::linear
{
    using polysolve::StiffnessMatrix;

    void Solver::apply_default_solver(json &rules, const std::string &prefix)
    {
        // set default wrt availability
        for (int i = 0; i < rules.size(); i++)
        {
            if (rules[i]["pointer"] == prefix + "/solver")
            {
                rules[i]["default"] = default_solver();
                rules[i]["options"] = available_solvers();
            }
            else if (rules[i]["pointer"] == prefix + "/precond")
            {
                rules[i]["default"] = default_precond();
                rules[i]["options"] = available_preconds();
            }
        }
    }

    void Solver::select_valid_solver(json &params, spdlog::logger &logger)
    {
        // if solver is an array, pick the first available
        const auto lin_solver_ptr = "/solver"_json_pointer;
        if (params.contains(lin_solver_ptr) && params[lin_solver_ptr].is_array())
        {
            const std::vector<std::string> solvers = params[lin_solver_ptr];
            const std::vector<std::string> available_solvers = Solver::available_solvers();
            std::string accepted_solver = "";
            for (const std::string &solver : solvers)
            {
                if (std::find(available_solvers.begin(), available_solvers.end(), solver) != available_solvers.end())
                {
                    accepted_solver = solver;
                    break;
                }
            }
            if (!accepted_solver.empty())
                logger.info("Solver {} is the highest priority available solver; using it.", accepted_solver);
            else
                logger.warn("No valid solver found in the list of specified solvers!");
            params[lin_solver_ptr] = accepted_solver;
        }

        // Fallback to default linear solver if the specified solver is invalid
        // NOTE: I do not know why .value() causes a segfault only on Windows
        // const bool fallback_solver = params.value("/enable_overwrite_solver"_json_pointer, false);
        const bool fallback_solver =
            params.contains("/enable_overwrite_solver"_json_pointer)
                ? params.at("/enable_overwrite_solver"_json_pointer).get<bool>()
                : false;
        if (fallback_solver)
        {
            const std::vector<std::string> ss = Solver::available_solvers();
            std::string s_json = "null";
            if (!params.contains(lin_solver_ptr) || !params[lin_solver_ptr].is_string()
                || std::find(ss.begin(), ss.end(), s_json = params[lin_solver_ptr].get<std::string>()) == ss.end())
            {
                logger.warn("Solver {} is invalid, falling back to {}", s_json, Solver::default_solver());
                params[lin_solver_ptr] = Solver::default_solver();
            }
        }
    }

    std::unique_ptr<Solver> Solver::create(const json &params_in, spdlog::logger &logger, const bool strict_validation)
    {
        json params = params_in; // mutable copy

        json rules;
        jse::JSE jse;

        jse.strict = strict_validation;
        const std::string input_spec = POLYSOLVE_LINEAR_SPEC;
        std::ifstream file(input_spec);

        if (file.is_open())
            file >> rules;
        else
            log_and_throw_error(logger, "unable to open {} rules", input_spec);

        apply_default_solver(rules);
        select_valid_solver(params, logger);

        const bool valid_input = jse.verify_json(params, rules);

        if (!valid_input)
            log_and_throw_error(logger, "invalid input json:\n{}", jse.log2str());

        params = jse.inject_defaults(params, rules);

        auto res = create(params["solver"], params["precond"]);
        res->set_parameters(params);

        return res;
    }

    ////////////////////////////////////////////////////////////////////////////////

#if EIGEN_VERSION_AT_LEAST(3, 3, 0)

// Magic macro because C++ has no introspection
#define ENUMERATE_PRECOND(HelperFunctor, SolverType, default_precond, precond, name)                                \
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
                                                           default_precond>::type>(name);                           \
        }                                                                                                           \
    } while (0)

#else

    // Magic macro because C++ has no introspection
#define ENUMERATE_PRECOND(HelperFunctor, SolverType, default_precond, precond)                       \
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
                                                           default_precond>::type>();                \
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
            typename default_precond = Eigen::DiagonalPreconditioner<double>>
        struct PrecondHelper
        {
            static std::unique_ptr<Solver> create(const std::string &arg, const std::string &name)
            {
                ENUMERATE_PRECOND(MakeSolver, SolverType, default_precond, arg, name);
            }
        };

        template <
            template <class, int, class> class SolverType,
            typename default_precond = Eigen::DiagonalPreconditioner<double>>
        struct PrecondHelperSym
        {
            static std::unique_ptr<Solver> create(const std::string &arg, const std::string &name)
            {
                ENUMERATE_PRECOND(MakeSolverSym, SolverType, default_precond, arg, name);
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
#ifdef POLYSOLVE_WITH_TRILINOS
        }
        else if (solver == "Trilinos")
        {
            return std::make_unique<TrilinosSolver>();
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
#ifdef POLYSOLVE_WITH_TRILINOS
            "Trilinos",
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

    std::string Solver::default_solver()
    {
        // return "Eigen::BiCGSTAB";
#ifdef POLYSOLVE_WITH_PARDISO
        return "Pardiso";
#else
#ifdef POLYSOLVE_WITH_ACCELERATE
        return "Eigen::AccelerateLDLT";
#else
#ifdef POLYSOLVE_WITH_HYPRE
        return "Hypre";
#else
        return "Eigen::BiCGSTAB";
#endif
#endif
#endif
    }

    // -----------------------------------------------------------------------------

    // List available preconditioners
    std::vector<std::string> Solver::available_preconds()
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

    std::string Solver::default_precond()
    {
        return "Eigen::DiagonalPreconditioner";
    }

    ////////////////////////////////////////////////////////////////////////////////

} // namespace polysolve::linear
