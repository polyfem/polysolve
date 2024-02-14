#ifdef POLYSOLVE_WITH_TRILINOS

////////////////////////////////////////////////////////////////////////////////
#include "TrilinosSolver.hpp"
#include <string>
#include <vector>
#include <unsupported/Eigen/SparseExtra>
/////////////////////////////////s///////////////////////////////////////////////

namespace polysolve::linear
{
    namespace{
    ////////////////////////////////////////////////////////////////
    // int rigid_body_mode(int ndim, const std::vector<double> &coo, std::vector<double> &B, bool transpose = true) {

    //     size_t n = coo.size();
    //     int nmodes = (ndim == 2 ? 3 : 6);
    //     B.resize(n * nmodes, 0.0);

    //     const int stride1 = transpose ? 1 : nmodes;
    //     const int stride2 = transpose ? n : 1;
    //     // int stride1 = nmodes;
    //     // int stride2 = 1;

    //     double sn = 1 / sqrt(n);

    //     if (ndim == 2) {
    //         for(size_t i = 0; i < n; ++i) {
    //             size_t nod = i / ndim;
    //             size_t dim = i % ndim;

    //             double x = coo[nod * 2 + 0];
    //             double y = coo[nod * 2 + 1];

    //             // Translation
    //             B[i * stride1 + dim * stride2] = sn;

    //             // Rotation
    //             switch(dim) {
    //                 case 0:
    //                     B[i * stride1 + 2 * stride2] = -y;
    //                     break;
    //                 case 1:
    //                     B[i * stride1 + 2 * stride2] = x;
    //                     break;
    //             }
    //         }
    //     } else if (ndim == 3) {
    //         for(size_t i = 0; i < n; ++i) {
    //             size_t nod = i / ndim;
    //             size_t dim = i % ndim;

    //             double x = coo[nod * 3 + 0];
    //             double y = coo[nod * 3 + 1];
    //             double z = coo[nod * 3 + 2];

    //             // Translation
    //             B[i * stride1 + dim * stride2] = sn;

    //             // Rotation
    //             switch(dim) {
    //                 case 0:
    //                     B[i * stride1 + 5 * stride2] = -y;
    //                     B[i * stride1 + 4 * stride2] = z;
    //                     break;
    //                 case 1:
    //                     B[i * stride1 + 5 * stride2] = x;
    //                     B[i * stride1 + 3 * stride2] = -z;
    //                     break;
    //                 case 2:
    //                     B[i * stride1 + 3 * stride2] =  y;
    //                     B[i * stride1 + 4 * stride2] = -x;
    //                     break;
    //             }
    //         }
    //     }

    //    // Orthonormalization
    //     std::array<double, 6> dot;
    //     for(int i = ndim; i < nmodes; ++i) {
    //         std::fill(dot.begin(), dot.end(), 0.0);
    //         for(size_t j = 0; j < n; ++j) {
    //             for(int k = 0; k < i; ++k)
    //                 dot[k] += B[j * stride1 + k * stride2] * B[j * stride1 + i * stride2];
    //         }
    //         double s = 0.0;
    //         for(size_t j = 0; j < n; ++j) {
    //             for(int k = 0; k < i; ++k)
    //                 B[j * stride1 + i * stride2] -= dot[k] * B[j * stride1 + k * stride2];
    //             s += B[j * stride1 + i * stride2] * B[j * stride1 + i * stride2];
    //         }
    //         s = sqrt(s);
    //         for(size_t j = 0; j < n; ++j)
    //             B[j * stride1 + i * stride2] /= s;
    //     }
    //     return nmodes;
    // }
    // Produce a vector of rigid body modes

    }
    TrilinosSolver::TrilinosSolver()
    {
        precond_num_ = 0;
#ifdef HAVE_MPI
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
#else
     CommPtr=new Epetra_SerialComm;
#endif
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
            if (params["Trilinos"].contains("is_nullspace"))
            {
                is_nullspace_ = params["Trilinos"]["is_nullspace"];
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


    void TrilinosSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        int output=10; //how often to print residual history
        Teuchos::ParameterList MLList;
        TrilinosML_SetDefaultOptions(MLList);
        MLList.set("PDE equations",numPDEs);
        
        //Set null space

        // if (true)
        // {
        // int n=test_vertices.rows();
        // int NRbm=0;
        // int NscalarDof=0;
        
        // if (numPDEs==2)
        // {
        //     NRbm=3;
        //     rbm=new double[n*(NRbm+NscalarDof)*(numPDEs+NscalarDof)];
        //     std::vector<double> z_coord(n,0);
        //     ML_Coord2RBM(n,test_vertices.col(0).data(),test_vertices.col(1).data(),z_coord.data(),rbm,numPDEs,NscalarDof);
        // }
        // else
        // {
        //     NRbm=6;
        //     rbm=new double[n*(NRbm+NscalarDof)*(numPDEs+NscalarDof)];
        //     ML_Coord2RBM(n,test_vertices.col(0).data(),test_vertices.col(1).data(),test_vertices.col(2).data(),rbm,numPDEs,NscalarDof);
        // }

        // MLList.set("null space: vectors",rbm);
        // MLList.set("null space: dimension", NRbm);        
        // MLList.set("null space: type", "pre-computed");
        // MLList.set("aggregation: threshold",0.00);
        // }

        ///////////////////////////////////////////////////////////////////////

        if (is_nullspace_)
        {
            if (test_vertices.cols()==3)
            {
                reduced_vertices=remove_boundary_vertices(test_vertices,test_boundary_nodes);
                MLList.set("null space: type","elasticity from coordinates");
                MLList.set("x-coordinates", reduced_vertices.col(0).data());
                MLList.set("y-coordinates", reduced_vertices.col(1).data());
                MLList.set("z-coordinates", reduced_vertices.col(2).data());
                MLList.set("aggregation: threshold",0.00);
            }
            if (test_vertices.cols()==2)
            {
                reduced_vertices=remove_boundary_vertices(test_vertices,test_boundary_nodes);
                MLList.set("null space: type","elasticity from coordinates");
                MLList.set("x-coordinates", reduced_vertices.col(0).data());
                MLList.set("y-coordinates", reduced_vertices.col(1).data());
                MLList.set("aggregation: threshold",0.00);
            }           

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
#ifdef HAVE_MPI
        MPI_Finalize() ;
#endif
    }
}

#endif
