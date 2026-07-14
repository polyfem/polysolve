#include "CatamariSolver.hpp"

#include <stdexcept>

namespace polysolve::linear
{

#if MESHFEM_WITH_CATAMARI
    CatamariSolver::CatamariSolver() = default;

    MeshFEM::CatamariFactorizer::OrderingMethod CatamariSolver::ordering_from_string(const std::string &ordering)
    {
        using OrderingMethod = MeshFEM::CatamariFactorizer::OrderingMethod;

        if (ordering == "Catamari")             return OrderingMethod::Catamari;
        if (ordering == "CholmodNesdis")        return OrderingMethod::CholmodNesdis;
        if (ordering == "Nesdis")               return OrderingMethod::CholmodNesdis;
        if (ordering == "Metis")                return OrderingMethod::Metis;
        if (ordering == "AMD")                  return OrderingMethod::AMD;
        if (ordering == "Adaptive")             return OrderingMethod::Adaptive;
        if (ordering == "Scotch")               return OrderingMethod::Scotch;
        if (ordering == "AccelerateMetis")      return OrderingMethod::AccelerateMetis;
        if (ordering == "PardisoMetis")         return OrderingMethod::PardisoMetis;
        if (ordering == "PardisoParallelMetis") return OrderingMethod::PardisoParallelMetis;

        throw std::runtime_error("Unknown Catamari ordering method: " + ordering);
    }

    void CatamariSolver::set_parameters(const json &params)
    {
        if (!params.contains("Catamari")) return;
        const json &catamari_params = params["Catamari"];
        if (catamari_params.contains("ordering"))         factorizer_.orderingMethod = ordering_from_string(catamari_params["ordering"].get<std::string>());
        if (catamari_params.contains("use_left_looking")) factorizer_.setUseLeftLooking(catamari_params["use_left_looking"].get<bool>());
        if (catamari_params.contains("use_block_accel"))  factorizer_.setUseBlockAccel(catamari_params["use_block_accel"].get<bool>());
    }

    void CatamariSolver::get_info(json &params) const
    {
        params["solver"] = name();
        params["reduced_size"] = reduced_size_;
        params["full_size"] = full_size_;
        if (symbolic_factorization_) {
            params["factor_nnz"] = factorizer_.getFactorNNZ();
            params["flop_estimate"] = factorizer_.getFlopEstimate();
        }
    }

    void CatamariSolver::analyze_pattern(const Hessian &H, const int)
    {
        const HessianType &A = H.as<HessianType>();
        factorizer_.factorizeSymbolic(*A.H, A.fixedVars());
        reduced_size_ = static_cast<Eigen::Index>(factorizer_.n_reduced());
        full_size_ = static_cast<Eigen::Index>(factorizer_.n());
    }

    void CatamariSolver::factorize(const Hessian &H)
    {
        const HessianType &A = H.as<HessianType>();
        if (!factorizer_.hasFactorization(FactorizationType::Symbolic) || factorizer_.wantsSymbolicFactorizationRecompute())
            analyze_pattern(H, 0);
        factorizer_.factorizeNumeric(*A.H);
    }

    void CatamariSolver::solve(const Ref<const VectorXd> rhs, Ref<VectorXd> result)
    {
        if (   rhs.size() != reduced_size_) throw std::runtime_error("[Catamari] Incorrect RHS size: expected "    + std::to_string(reduced_size_) + ", got " + std::to_string(rhs.size()));
        if (result.size() != reduced_size_) throw std::runtime_error("[Catamari] Incorrect result size: expected " + std::to_string(reduced_size_) + ", got " + std::to_string(result.size()));

        factorizer_.solveRawReduced(rhs.data(), result.data());
    }
#endif

} // namespace polysolve::linear
