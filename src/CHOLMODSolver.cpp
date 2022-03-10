#include "CHOLMODSolver.hpp"
#include <Eigen/CholmodSupport>
#include <iostream>

namespace polysolve
{

    cholmod_sparse eigen2cholmod(StiffnessMatrixL &mat)
    {
        cholmod_sparse res;
        res.nzmax   = mat.nonZeros();
        res.nrow    = mat.rows();
        res.ncol    = mat.cols();
        res.p       = mat.outerIndexPtr();
        res.i       = mat.innerIndexPtr();
        res.x       = mat.valuePtr();
        res.z       = 0;
        res.sorted  = 1;
        if(mat.isCompressed())
        {
            res.packed  = 1;
            res.nz = 0;
        }
        else
        {
            res.packed  = 0;
            res.nz = mat.innerNonZeroPtr();
        }

        res.dtype = CHOLMOD_DOUBLE;

        res.itype = CHOLMOD_LONG;
        res.stype = 1;
        res.xtype = CHOLMOD_REAL;
  
        return res;
    }

    cholmod_dense eigen2cholmod(Eigen::VectorXd &mat)
    {
        cholmod_dense res;
        res.nrow   = mat.size();
        res.ncol   = 1;
        res.nzmax  = res.nrow * res.ncol;
        res.d      = mat.derived().size();
        res.x      = (void*)(mat.derived().data());
        res.z      = 0;
        res.xtype = CHOLMOD_REAL;
        res.dtype   = 0;
  
        return res;
    }
    ////////////////////////////////////////////////////////////////////////////////

    CHOLMODSolver::CHOLMODSolver()
    {
        cm = (cholmod_common*)malloc(sizeof(cholmod_common));
        cholmod_l_start (cm);
        L = NULL;
        
        cm->useGPU = 1;
        cm->maxGpuMemBytes = 2e9;
        cm->print = 4;
        cm->supernodal = CHOLMOD_SUPERNODAL;
    }

    // void CHOLMODSolver::getInfo(json &params) const
    // {
    //     params["num_iterations"] = num_iterations;
    //     params["final_res_norm"] = final_res_norm;
    // }

    void CHOLMODSolver::analyzePattern(StiffnessMatrixL &Ain) 
    {   
        if(!Ain.isCompressed()) {
            Ain.makeCompressed();
        }
        A = eigen2cholmod(Ain);
        L = cholmod_l_analyze (&A, cm);
    }

    void CHOLMODSolver::factorize(StiffnessMatrixL &Ain)
    {
        cholmod_l_factorize (&A, L, cm);
    }

    void CHOLMODSolver::solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result)
    {
        b = eigen2cholmod(rhs);
        x = cholmod_l_solve (CHOLMOD_A, L, &b, cm);

        memcpy(result.data(), x->x, result.size() * sizeof(result[0]));
    }

    ////////////////////////////////////////////////////////////////////////////////

    CHOLMODSolver::~CHOLMODSolver()
    {
        cholmod_l_gpu_stats(cm);
        cholmod_l_free_factor (&L, cm);
        cholmod_l_free_dense (&x, cm);
    }

} // namespace polysolve
