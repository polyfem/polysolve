#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <nlohmann/json.hpp>
#include <MeshFEM/newton_optimizer/NewtonHessian.hh>

#include <variant>

namespace polysolve
{

#ifdef POLYSOLVE_LARGE_INDEX
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t> StiffnessMatrix;
#else
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor> StiffnessMatrix;
#endif

    using json = nlohmann::json;

    struct Hessian {
        using Variant = std::variant<StiffnessMatrix, Eigen::MatrixXd, NewtonHessian>;
        Variant evaluated_hessian;

        template<typename T>
        explicit Hessian(std::in_place_type_t<T>) : evaluated_hessian(std::in_place_type<T>) {}


        template<typename T>
        T &get() {
            T *H = std::get_if<T>(&evaluated_hessian);
            if (H) return *H;
            throw std::runtime_error("Hessian does not hold the requested type");
        }

        Eigen::VectorXd operator*(const Eigen::VectorXd &v) const {
            if (const auto *H = std::get_if<StiffnessMatrix>(&evaluated_hessian)) return *H * v;
            if (const auto *H = std::get_if<Eigen::MatrixXd>(&evaluated_hessian))  return *H * v;
            if (const auto *H = std::get_if<NewtonHessian>(&evaluated_hessian))    return H->apply(v);
            throw std::runtime_error("Unknown Hessian type");
        }
    

        template<typename T>
        const T &get() const {
            const T *H = std::get_if<T>(&evaluated_hessian);
            if (H) return *H;

            converted_hessian = Variant(std::in_place_type<T>);
            auto &H_c = std::get<T>(converted_hessian);

            return H_c;
        }

        Eigen::Index rows() const {
            if (const auto *H = std::get_if<NewtonHessian>(&converted_hessian))    return static_cast<Eigen::Index>(H->numVars());
            else if (const auto *H = std::get_if<StiffnessMatrix>(&converted_hessian)) return H->rows();
            else if (const auto *H = std::get_if<Eigen::MatrixXd>(&converted_hessian)) return H->rows();
            throw std::runtime_error("Unknown Hessian type");
        }

        
    private:
        mutable Variant converted_hessian;
    };

} // namespace polysolve
