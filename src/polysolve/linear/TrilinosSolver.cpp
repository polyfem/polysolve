#ifdef POLYSOLVE_WITH_TRILINOS

////////////////////////////////////////////////////////////////////////////////
#include "TrilinosSolver.hpp"
#include <string>
#include <vector>
#include <unsupported/Eigen/SparseExtra>
////////////////////////////////////////////////////////////////////////////////

namespace polysolve::linear
{
    TrilinosSolver::TrilinosSolver()
    {
        precond_num_ = 0;
        int done_already;
        MPI_Initialized(&done_already);
        if (!done_already)
        {
            /* Initialize MPI */
            int argc = 1;
            char name[] = "";
            char *argv[] = {name};
            char **argvv = &argv[0];
            MPI_Init(&argc, &argvv);
            CommPtr = new Epetra_MpiComm(MPI_COMM_WORLD);
        }
    }

    ////////////////////////////////////////////////////////////////
    void TrilinosSolver::set_parameters(const json &params)
    {
        if (params.contains("Trilinos"))
        {
            if (params["Trilinos"].contains("block_size"))
            {
                if (params["Trilinos"]["block_size"] == 2 || params["Trilinos"]["block_size"] == 3)
                {
                    numPDEs = params["Trilinos"]["block_size"];
                }
            }
            if (params["Trilinos"].contains("max_iter"))
            {
                max_iter_ = params["Trilinos"]["max_iter"];
            }
            if (params["Trilinos"].contains("tolerance"))
            {
                conv_tol_ = params["Trilinos"]["tolerance"];
            }
        }
    }

    /////////////////////////////////////////////////
    void TrilinosSolver::get_info(json &params) const
    {
        params["num_iterations"] = iterations_;
        params["final_res_norm"] = residual_error_;
    }

    /////////////////////////////////////////////////
    void TrilinosSolver::factorize(const StiffnessMatrix &Ain)
    {
        assert(precond_num_ > 0);
        // Eigen::saveMarket(Ain,"/home/yiwei/matrix_struct/A_nonLinear.mtx");
        // Eigen::saveMarket(test_vertices,"/home/yiwei/matrix_struct/vec.mtx");
        Eigen::SparseMatrix<double,Eigen::RowMajor> Arow(Ain);
        int mypid = CommPtr->MyPID();
        int indexBase=0;
        int numGlobalElements = Arow.nonZeros();
        int numGlobalRows=Arow.rows();
        
        int numNodes= numGlobalRows /numPDEs;
        if ((numGlobalRows - numNodes * numPDEs) != 0 && !mypid){
            throw std::runtime_error("Number of matrix rows is not divisible by #dofs");
        }
        int numMyNodes;
        int nproc = CommPtr->NumProc();
        if (CommPtr->MyPID() < nproc-1) numMyNodes = numNodes / nproc;
        else numMyNodes = numNodes - (numNodes/nproc) * (nproc-1);
        delete rowMap;
        delete A;
        rowMap = new Epetra_Map(numGlobalRows,numMyNodes*numPDEs,indexBase,(*CommPtr));

        A = new Epetra_CrsMatrix(Copy,*rowMap,0); //Can allocate memory for each row in advance

        {
            int nnzs=0;
            for (int k=0 ; k < Arow.outerSize(); k++)
            {
                // std::cout<<Arow.innerVector(k).nonZeros()<<" ";
                int numEntries=Arow.outerIndexPtr()[k+1]-Arow.outerIndexPtr()[k];
                int* indices=Arow.innerIndexPtr () + nnzs;
                double* values=Arow.valuePtr () + nnzs;
                A->InsertGlobalValues(k,numEntries,values,indices);
                nnzs=nnzs+numEntries;
            }
        }
        A->FillComplete();
    }

    namespace
    {
        void TrilinosML_SetDefaultOptions(Teuchos::ParameterList &MLList)
        {
            std::string aggr_type="Uncoupled-MIS";
            double str_connect=0.08;
            ML_Epetra::SetDefaults("SA", MLList);
            MLList.set("aggregation: type",aggr_type); // Aggresive Method
            // MLList.set("aggregation: type","Uncoupled"); // Fixed size


            MLList.set("aggregation: threshold",str_connect);

            // Smoother Settings
            MLList.set("smoother: type","Chebyshev");
            MLList.set("smoother: sweeps", 5); //Chebyshev degree
            MLList.set("smoother: Chebyshev alpha",30.0);

            //Coarser Settings
            MLList.set("coarse: max size",1000);

            MLList.set("ML output",0);
        }
    }


    void TrilinosSolver::solve(const Eigen::Ref<const VectorXd> rhs, const Eigen::Ref<const MatrixXd> nullspace, Eigen::Ref<VectorXd> result)
    {
        int output=10; //how often to print residual history
        Teuchos::ParameterList MLList;
        TrilinosML_SetDefaultOptions(MLList);
        MLList.set("PDE equations",numPDEs);

        if (nullspace.cols()==3)
        {
            Eigen::VectorXd col1 = nullspace.col(0);
            int size1 = col1.size();
            double* arr1 = new double[size1];
            for(int i = 0; i < size1; ++i) {
                arr1[i] = col1(i);
            }
            Eigen::VectorXd col2 = nullspace.col(1);
            int size2 = col2.size();
            double* arr2 = new double[size2];
            for(int i = 0; i < size2; ++i) {
                arr2[i] = col2(i);
            }
            Eigen::VectorXd col3 = nullspace.col(2);
            int size3 = col3.size();
            double* arr3 = new double[size3];
            for(int i = 0; i < size3; ++i) {
                arr3[i] = col3(i);
            }
            MLList.set("null space: type","elasticity from coordinates");
            MLList.set("x-coordinates", arr1);  // nullspace.col(0).data()
            MLList.set("y-coordinates", arr2);  // nullspace.col(1).data()
            MLList.set("z-coordinates", arr3);  // nullspace.col(2).data()
            MLList.set("aggregation: threshold",0.00);
        }
        if (nullspace.cols()==2)
        {
            Eigen::VectorXd col1 = nullspace.col(0);
            int size1 = col1.size();
            double* arr1 = new double[size1];
            for(int i = 0; i < size1; ++i) {
                arr1[i] = col1(i);
            }
            Eigen::VectorXd col2 = nullspace.col(1);
            int size2 = col2.size();
            double* arr2 = new double[size2];
            for(int i = 0; i < size2; ++i) {
                arr2[i] = col2(i);
            }
            MLList.set("null space: type","elasticity from coordinates");
            MLList.set("x-coordinates", arr1);  // nullspace.col(0).data()
            MLList.set("y-coordinates", arr2);  // nullspace.col(1).data()
            MLList.set("aggregation: threshold",0.00);
        }

        delete MLPrec;
        MLPrec = new ML_Epetra::MultiLevelPreconditioner(*A, MLList);
        Epetra_Vector x(A->RowMap());
        Epetra_Vector b(A->RowMap());
        for (size_t i = 0; i < rhs.size(); i++)
        {
            x[i]=result[i];
            b[i]=rhs[i];
        }

        Epetra_LinearProblem Problem(A,&x,&b);
        AztecOO solver(Problem);
        solver.SetAztecOption(AZ_solver, AZ_cg);
        solver.SetPrecOperator(MLPrec);
        solver.SetAztecOption(AZ_output, AZ_last);

       int status= solver.Iterate(max_iter_, conv_tol_ );
        if (status!=0 && status!=-3)
        {
            throw std::runtime_error("Early termination, not SPD");
        }

        if (CommPtr->MyPID() == 0)
        {
            residual_error_=solver.ScaledResidual ();
            iterations_=solver.NumIters();
        }

        for (size_t i = 0; i < rhs.size(); i++)
        {
            result[i]=x[i];
        }

    }

    void TrilinosSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        int output=10; //how often to print residual history
        Teuchos::ParameterList MLList;
        TrilinosML_SetDefaultOptions(MLList);
        MLList.set("PDE equations",numPDEs);

        delete MLPrec;
        MLPrec = new ML_Epetra::MultiLevelPreconditioner(*A, MLList);
        Epetra_Vector x(A->RowMap());
        Epetra_Vector b(A->RowMap());
        for (size_t i = 0; i < rhs.size(); i++)
        {
            x[i]=result[i];
            b[i]=rhs[i];
        }
        // std::cout<<"x[0] "<<x[0]<<std::endl;
        // std::cout<<"b[0] "<<b[0]<<std::endl;

        Epetra_LinearProblem Problem(A,&x,&b);
        AztecOO solver(Problem);
        solver.SetAztecOption(AZ_solver, AZ_cg);
        solver.SetPrecOperator(MLPrec);
        solver.SetAztecOption(AZ_output, AZ_last);

       int status= solver.Iterate(max_iter_, conv_tol_ );
        if (status!=0 && status!=-3)
        {
            throw std::runtime_error("Early termination, not SPD");
        }
        

        //Calculate a final residual
        // Epetra_Vector workvec(A->RowMap());
        // double mynorm;
        // A->Multiply(false,x,workvec);
        // workvec.Update(1.0,b,-1.0);
        // b.Norm2(&mynorm);
        // workvec.Scale(1./mynorm);
        // workvec.Norm2(&mynorm);
        if (CommPtr->MyPID() == 0)
        {

            // std::cout<<"Max iterations "<<max_iter_<<std::endl;
            // std::cout<<"Trilinos ScaleResidual is is "<<solver.TrueResidual ()<<std::endl;
            // std::cout<<"Trilinos ScaleResidual is "<<solver.ScaledResidual ()<<std::endl;
            // std::cout<<"Iterations are "<<solver.NumIters()<<std::endl;
            residual_error_=solver.ScaledResidual ();
            iterations_=solver.NumIters();
        }
        
        // if (iterations_>175)
        // {
        //    exit();
        // }
        


        for (size_t i = 0; i < rhs.size(); i++)
        {
            result[i]=x[i];
        }
   
    }

    TrilinosSolver:: ~TrilinosSolver()
    {
        delete A;
        delete rowMap;
        delete MLPrec;   
        MPI_Finalize() ;
    }
}

#endif
