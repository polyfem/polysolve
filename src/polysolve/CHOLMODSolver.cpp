#ifdef POLYSOLVE_WITH_CHOLMOD

#include "CHOLMODSolver.hpp"
#include <Eigen/CholmodSupport>

#include <stdio.h>
#include <iostream>
#include <memory>

namespace polysolve
{
    namespace
    {
        cholmod_sparse eigen2cholmod(const StiffnessMatrix &mat)
        {
            cholmod_sparse res;
            res.nzmax = mat.nonZeros();
            res.nrow = mat.rows();
            res.ncol = mat.cols();

            memcpy(res.p, mat.outerIndexPtr(), sizeof(mat.outerIndexPtr()));
            memcpy(res.i, mat.innerIndexPtr(), sizeof(mat.innerIndexPtr()));
            memcpy(res.x, mat.valuePtr(), sizeof(mat.valuePtr()));

            res.z = 0;
            res.sorted = 1;
            // if (mat.isCompressed())
            {
                assert(mat.isCompressed());
                res.packed = 1;
                res.nz = 0;
            }
            // else
            // {
            //     res.packed = 0;
            //     res.nz = mat.innerNonZeroPtr();
            // }

            res.dtype = CHOLMOD_DOUBLE;

            res.itype = CHOLMOD_LONG;
            res.stype = 1;
            res.xtype = CHOLMOD_REAL;

            return res;
        }
    } // namespace

    cholmod_dense eigen2cholmod(Eigen::VectorXd &mat)
    {
        cholmod_dense res;
        res.nrow = mat.size();
        res.ncol = 1;
        res.nzmax = res.nrow * res.ncol;
        res.d = mat.derived().size();
        res.x = (void *)(mat.derived().data());
        res.z = 0;
        res.xtype = CHOLMOD_REAL;
        res.dtype = 0;

        return res;
    }
    ////////////////////////////////////////////////////////////////////////////////

    CHOLMODSolver::CHOLMODSolver()
    {
        cm = (cholmod_common *)malloc(sizeof(cholmod_common));
        cholmod_l_start(cm);
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

    void CHOLMODSolver::analyzePattern(const StiffnessMatrix &Ain, const int precond_num)
    {
        assert(Ain.isCompressed());

        A = eigen2cholmod(Ain);
        L = cholmod_l_analyze(&A, cm);
    }

    void CHOLMODSolver::factorize(const StiffnessMatrix &Ain)
    {
        cholmod_l_factorize(&A, L, cm);
    }

    void CHOLMODSolver::solve(const Ref<const VectorXd> rhs, Ref<VectorXd> result)
    {
        assert(result.size() == rhs.size());

        // b = eigen2cholmod(rhs); //TODO: fix me

        x = cholmod_l_solve(CHOLMOD_A, L, &b, cm);

        memcpy(result.data(), x->x, result.size() * sizeof(result[0]));
    }

    ////////////////////////////////////////////////////////////////////////////////

    CHOLMODSolver::~CHOLMODSolver()
    {
        // cholmod_l_gpu_stats(cm);
        cholmod_l_free_factor(&L, cm);
        cholmod_l_free_dense(&x, cm);
    }

} // namespace polysolve

#endif
