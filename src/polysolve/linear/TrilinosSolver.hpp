#pragma once

#ifdef POLYSOLVE_WITH_TRILINOS

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

#ifdef HAVE_MPI
#include "mpi.h"
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif
#include "Teuchos_CommandLineProcessor.hpp"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_LinearProblem.h"
#include "EpetraExt_BlockMapIn.h"
#include "EpetraExt_CrsMatrixIn.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_MultiVectorIn.h"
#include "AztecOO.h"

#include "ml_include.h"
#include "ml_MultiLevelPreconditioner.h"
#include "ml_epetra.h"
#include <fstream>

////////////////////////////////////////////////////////////////////////////////
//
// WARNING:
// The matrix is assumed to be in row-major format, since AMGCL assumes that the
// outer index is for row. If the matrix is symmetric, you are fine, because CSR
// and CSC are the same. If the matrix is not symmetric and you pass in a
// column-major matrix, the solver will actually solve A^T x = b.
//

namespace polysolve::linear
{

    class TrilinosSolver : public Solver
    {

    public:
        TrilinosSolver();
        ~TrilinosSolver();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(TrilinosSolver)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Retrieve memory information from Pardiso
        virtual void get_info(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override { precond_num_ = precond_num; }

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
        virtual void solve(const Ref<const VectorXd> b, const Ref<const MatrixXd> nullspace, Ref<VectorXd> x);
        
        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "Trilinos AztecOO and ML"; }

    protected:
        int numPDEs = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 1000;
        double conv_tol_ = 1e-8;
        size_t iterations_;
        double residual_error_;
        ML_Epetra::MultiLevelPreconditioner* MLPrec=NULL;
        Epetra_Map *rowMap=NULL;

    private:
        int precond_num_;
        Epetra_CrsMatrix *A=NULL;
#ifdef HAVE_MPI
        Epetra_MpiComm *CommPtr;
#else
        Epetra_SerialComm *CommPtr;
#endif
    };

} // namespace polysolve

#endif