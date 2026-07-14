#pragma once

#include "Solver.hpp"

#include <MeshFEMSparse/Solvers/CatamariFactorizer.hh>

namespace polysolve::linear
{

#if MESHFEM_WITH_CATAMARI
    class CatamariSolver final : public Solver
    {
    public:
        CatamariSolver();

        POLYSOLVE_DELETE_MOVE_COPY(CatamariSolver)

        void set_parameters(const json &params) override;
        void get_info(json &params) const override;

        void analyze_pattern(const Hessian &H, const int precond_num) override;
        void factorize(const Hessian &H) override;
        void solve(const Ref<const VectorXd> rhs, Ref<VectorXd> result) override;

        std::string name() const override { return "Catamari"; }

    private:
        using HessianType = polysolve::BCSCHessianWithFixedVars;
        using FactorizationType = MeshFEM::CholeskyFactorizerBase::FactorizationType;

        static MeshFEM::CatamariFactorizer::OrderingMethod ordering_from_string(const std::string &ordering);

        MeshFEM::CatamariFactorizer factorizer_;
        Eigen::Index reduced_size_ = -1;
        Eigen::Index full_size_ = -1;
        bool symbolic_factorization_ = false;
    };
#endif

} // namespace polysolve::linear
