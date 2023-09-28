#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <nlohmann/json.hpp>

namespace polysolve
{

#ifdef POLYSOLVE_LARGE_INDEX
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t> StiffnessMatrix;
#else
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor> StiffnessMatrix;
#endif

    using json = nlohmann::json;

// TODO
#define POLYSOLVE_SCOPED_TIMER(...) //

} // namespace polysolve