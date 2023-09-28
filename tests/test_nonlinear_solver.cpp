//////////////////////////////////////////////////////////////////////////
#include <polysolve/nonlinear/Problem.hpp>
#include <polysolve/nonlinear/SparseNewton.hpp>

#include <catch2/catch.hpp>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>
//////////////////////////////////////////////////////////////////////////

using namespace polysolve::nonlinear;

class TestProblem : public Problem
{
    double value(const TVector &x) override
    {
        return 0;
    }
    void gradient(const TVector &x, TVector &gradv) override
    {
        gradv.resize(10);
    }
    void hessian(const TVector &x, THessian &hessian) override
    {
        hessian.resize(10, 10);
    }
};

TEST_CASE("instantiation", "[solver]")
{
    TestProblem prob;

    SparseNewton<TestProblem> sn;
}