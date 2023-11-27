//////////////////////////////////////////////////////////////////////////
#include <polysolve/JSONUtils.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <catch2/catch.hpp>
#include <iostream>
#include <chrono>
//////////////////////////////////////////////////////////////////////////

using namespace polysolve;

TEST_CASE("json_to_eigen", "[json]")
{
    json input = json::parse(R"(
        [
            [1, 2, 3],
            [[1, 2, 3]],
            [[1], [2], [3]],
            [[1, 2], [3, 4]]
        ]
    )");

    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> a;
    Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor, 1, 3> b;
    Eigen::MatrixXd c;

    // 1D json array can be assigned to a row vector, col vector, or a matrix
    a = input[0];
    REQUIRE(a.rows() == 3);

    b = input[0];
    REQUIRE(b.cols() == 3);

    c = input[0];
    REQUIRE(c.cols() == 3);
    REQUIRE(c.rows() == 1);

    // 2D json array of size 1x3 can be assigned to a row vector or a matrix
    b = input[1];
    REQUIRE(b.cols() == 3);

    c = input[1];
    REQUIRE(c.cols() == 3); 
    REQUIRE(c.rows() == 1);

    // 2D json array of size 3x1 can be assigned to a col vector or a matrix
    a = input[2];
    REQUIRE(a.rows() == 3);

    c = input[2];
    REQUIRE(c.rows() == 3);
    REQUIRE(c.cols() == 1);

    // 2D json array of size 2x2 can only be assigned to a matrix
    c = input[3];
    REQUIRE(c.rows() == 2);
    REQUIRE(c.cols() == 2);
}

TEST_CASE("eigen_to_json", "[json]")
{
    json args;

    // a 3x3 matrix is transformed to a 2D json array
    Eigen::Matrix3d A;
    A.setZero();

    args = A;
    REQUIRE(args.size() == 3);
    REQUIRE(args[0].size() == 3);

    // a col vector is transformed to a 1D json array
    Eigen::VectorXd B;
    B.setZero(5);

    args = B;
    REQUIRE(args.size() == 5);

    // a row vector is transformed to a 1D json array
    Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> C;
    C.setZero(5);

    args = C;
    REQUIRE(args.size() == 5);
}
