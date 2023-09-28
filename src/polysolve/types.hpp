#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace polysolve
{

#ifdef POLYSOLVE_LARGE_INDEX
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t> StiffnessMatrix;
#else
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor> StiffnessMatrix;
#endif

} // namespace polysolve