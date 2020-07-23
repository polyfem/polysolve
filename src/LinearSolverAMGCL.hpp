#pragma once

#ifdef POLYSOLVE_WITH_AMGCL
// #define POLYSOLVE_AMGCL_V2
// #define POLYSOLVE_AMGCL_SIMPLE
#define POLYSOLVE_AMGCL_DUMMY_PRECOND

////////////////////////////////////////////////////////////////////////////////
#include <polysolve/LinearSolver.hpp>

#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/fgmres.hpp>

#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/aggregation.hpp>

#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/iluk.hpp>
#include <amgcl/relaxation/ilut.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>

#include <amgcl/preconditioner/dummy.hpp>
#include <amgcl/preconditioner/schur_pressure_correction.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>


#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

    ////////////////////////////////////////////////////////////////////////////////
    //
    //

    namespace polysolve {

    class LinearSolverAMGCL : public LinearSolver {

    public:
        LinearSolverAMGCL();
        ~LinearSolverAMGCL();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(LinearSolverAMGCL)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void setParameters(const json &params) override;

        // Retrieve memory information from Pardiso
        virtual void getInfo(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyzePattern(const StiffnessMatrix &A, const int precond_num) override;

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &) override { }

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;

        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "AMGCL"; }

    private:
        // typedef amgcl::backend::eigen<double> Backend;
        typedef amgcl::backend::builtin<double> Backend;

        // typedef amgcl::make_solver<
        // 	amgcl::amg<
        // 		Backend,
        // 		amgcl::coarsening::aggregation,
        // 		amgcl::relaxation::gauss_seidel>,
        // 	amgcl::solver::cg<Backend>> Solver;

        // typedef amgcl::make_solver<
        // 	// Use AMG as preconditioner:
        // 	amgcl::amg<
        // 		Backend,
        // 		amgcl::coarsening::smoothed_aggregation,
        // 		amgcl::relaxation::spai0>,
        // 	// And BiCGStab as iterative solver:
        // 	amgcl::solver::bicgstab<Backend>>
        // 	Solver;

#ifdef POLYSOLVE_AMGCL_SIMPLE
        // Use AMG as preconditioner:
        typedef amgcl::make_solver<
        	// Use AMG as preconditioner:
        	amgcl::amg<
        		Backend,
        		amgcl::coarsening::smoothed_aggregation,
        		amgcl::relaxation::gauss_seidel>,
        	// And CG as iterative solver:
        	amgcl::solver::cg<Backend>>
        	Solver;
#endif



#ifdef POLYSOLVE_AMGCL_DUMMY_PRECOND
        typedef amgcl::make_solver<
            amgcl::preconditioner::schur_pressure_correction<
                amgcl::make_solver<
                    amgcl::amg<
                        Backend,
                        amgcl::coarsening::smoothed_aggregation,
                        amgcl::relaxation::ilu0>,
                    amgcl::solver::bicgstab<Backend>>,
                amgcl::make_solver<
                    amgcl::preconditioner::dummy<Backend>,
                    amgcl::solver::bicgstab<Backend>>>,
            amgcl::solver::fgmres<Backend>>
            Solver;
#endif
#ifdef POLYSOLVE_AMGCL_V1
        typedef amgcl::make_solver<
            amgcl::relaxation::as_preconditioner<Backend,
                                                 amgcl::relaxation::iluk>,
            amgcl::solver::cg<Backend>>
            Solver;
#endif

#ifdef POLYSOLVE_AMGCL_V2
        typedef amgcl::make_solver<
            amgcl::preconditioner::schur_pressure_correction<amgcl::make_solver< // Solver for (8b)
                                                                 amgcl::amg<
                                                                     Backend,
                                                                     amgcl::coarsening::aggregation,
                                                                     amgcl::relaxation::ilut>,
                                                                 amgcl::solver::cg<Backend>>,
                                                             amgcl::make_solver< // Solver for (8a)
                                                                 amgcl::relaxation::as_preconditioner<Backend,
                                                                                                      amgcl::relaxation::spai0>,
                                                                 amgcl::solver::cg<Backend>>>,
            amgcl::solver::fgmres<Backend>>
            Solver;
#endif

        Solver *solver_ = nullptr;
        Solver::params params_;
        // StiffnessMatrix mat;

        std::vector<int> ia_, ja_;
        std::vector<double> a_;

        size_t iterations_;
        double residual_error_;
    };

} // namespace polysolve

#endif
