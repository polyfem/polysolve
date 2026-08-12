// Support for various Hessian representations and conversions between them.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <MeshFEMSparse/BlockCSCHessian.hh>

#include <optional>
#include <variant>
#include <type_traits>
#include <functional>

namespace polysolve
{
    // The Catamari solver can implement pin constraints more efficiently than
    // performing an explicit slicing operation. We therefore provide this
    // wrapper type to enable the pinned variables to be communicated directly
    // to the solver that support them.
    //
    // When fixed variables are present, solves are done in the reduced space
    // (i.e., the user's right-hand side and the returned solution
    // will both be of size `rows() - nfv`, where `nfv` is the number of
    // unique indices in `fixedVars). This differs from MeshFEMSparse's
    // convention of working with full-space RHS and solution vectors.
    struct BCSCHessianWithFixedVars {
        using BCSCHessian = MeshFEM::BlockCSCHessianBase;
        std::unique_ptr<BCSCHessian> H;
        double reg_weight = 0.0; // Multiple of the identity to add to the Hessian during factorization (used, e.g., by `RegularizedNewton`)

        BCSCHessianWithFixedVars() = default;
        BCSCHessianWithFixedVars(std::unique_ptr<BCSCHessian> &&H_, const std::vector<size_t> &fixedVars_ = {})
            : H(std::move(H_)), m_fixedVars(fixedVars_) { }

        static BCSCHessianWithFixedVars fromEigen(const StiffnessMatrix &H_eigen) {
            return BCSCHessianWithFixedVars{BCSCHessian::fromEigen(H_eigen)};
        }

        StiffnessMatrix toEigen() const { return H->template toEigen<StiffnessMatrix::StorageIndex>(/* upperTriangleOnly = */ false, m_fixedVars); }

        size_t full_size() const { return H->numScalarCols(); }
        // WARNING: assumes all fixedVars are unique and in-bounds!
        // We currently don't validate this here because it is already validated within PolyFEM.
        size_t reduced_size() const { return full_size() - m_fixedVars.size(); }

        // Basic "duck typing" compatibility with Eigen matrix types.
        size_t rows() const { return reduced_size(); }
        size_t cols() const { return reduced_size(); }
        Eigen::VectorXd operator*(const Eigen::VectorXd &v) const {
            if (m_fixedVars.empty()) return H->apply(v);

            m_buildFixedMask();
            if (size_t(v.size()) != reduced_size()) throw std::runtime_error("BCSCHessianWithFixedVars::operator*: incorrect reduced vector size");

            Eigen::VectorXd v_full = Eigen::VectorXd::Zero(full_size());
            for (size_t i = 0, r = 0; i < full_size(); ++i)
                if (!m_isFixed[i]) v_full[i] = v[r++];

            Eigen::VectorXd result_full = H->apply(v_full);
            if (reg_weight != 0) result_full += reg_weight * v_full;

            Eigen::VectorXd result(reduced_size());
            for (size_t i = 0, r = 0; i < full_size(); ++i)
                if (!m_isFixed[i]) result[r++] = result_full[i];
            return result;
        }

        const std::vector<size_t> &fixedVars() const { return m_fixedVars; }

        template<typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
        void setFixedVars(const std::vector<T> &fv) {
            m_fixedVars = std::vector<size_t>(fv.begin(), fv.end()); // Potential integer type conversion...
            m_isFixed.clear();
        }

    private:
        void m_buildFixedMask() const {
            if (m_isFixed.size() == full_size()) return;
            m_isFixed.resize(full_size(), false);
            for (size_t i : m_fixedVars) {
                if (i >= full_size()) throw std::runtime_error("BCSCHessianWithFixedVars::buildFixedMask: fixed variable index out of bounds");
                if (m_isFixed[i])     throw std::runtime_error("BCSCHessianWithFixedVars::buildFixedMask: duplicate fixed variable index");
                m_isFixed[i] = true;
            }
        }

        std::vector<size_t> m_fixedVars;
        mutable std::vector<bool> m_isFixed;
    };

    struct HessianConversion {
        using DenseHessian = Eigen::MatrixXd;

        template <typename T>
        struct type_tag { using type = T; };

        // Note: a conversion function must exist for all pairwise conversions
        // in order for the std::visit` call to be well formed. Only a few
        // conversions are actually supported/used in practice and should be
        // implemented as type-specific overloads. Identity conversions are
        // unnecessary since the initial `get_if<T>` attempt will succeed when
        // source and target conversion types coincide.
        template<class T_dst, class T_src> static T_dst convert(type_tag<T_dst>, const T_src &/* H */) {
            throw std::runtime_error("Hessian conversion not supported from " + std::string(typeid(T_src).name()) + " to " + std::string(typeid(T_dst).name()));
        }

        // Supported Hessian conversions
        static BCSCHessianWithFixedVars convert(type_tag<BCSCHessianWithFixedVars>, const StiffnessMatrix &H) { return BCSCHessianWithFixedVars::fromEigen(H); } // TODO: this conversion can be avoided once MeshFEMSparse updates its `CatamariFactorizer` wrapper to support Eigen sparse matrices natively.
        static StiffnessMatrix          convert(type_tag<StiffnessMatrix>, const BCSCHessianWithFixedVars &H) { StiffnessMatrix K = H.toEigen(); if (H.reg_weight != 0.0) { K.diagonal().array() += H.reg_weight; } return K; }
        static DenseHessian             convert(type_tag<DenseHessian>,    const BCSCHessianWithFixedVars &H) { DenseHessian    D = H.toEigen(); if (H.reg_weight != 0.0) { D.diagonal().array() += H.reg_weight; } return D; } // TODO: Avoid double-conversion?
        static DenseHessian             convert(type_tag<DenseHessian>,    const          StiffnessMatrix &H) { return DenseHessian(H); }
    };

    // Stores one of a number of possible Hessian types and facilitates
    // conversion into other types.
    // This is intended as a lightweight adapter class to interface between Hessian
    // evaluation routines, which evaluate in their preferred format, and
    // solver routines that accept only certain formats.
    //
    // Since the `Problem` API takes the output `Hessian` by reference rather
    // than returning it, this type is forced to start out in an "empty" state.
    template<class... HessianTypes>
    struct VariantTypes {
        using Variant = std::variant<HessianTypes..., std::reference_wrapper<const HessianTypes>...>;
        using CacheTuple = std::tuple<std::optional<HessianTypes>...>;

        template<typename T>
        static constexpr bool contains = (std::is_same_v<T, HessianTypes> || ...);
    };

    struct Hessian {
        using DenseHessian = HessianConversion::DenseHessian;
        using Types = VariantTypes<StiffnessMatrix, DenseHessian, BCSCHessianWithFixedVars>;
        using Variant = typename Types::Variant;

        Hessian() = default;

        // Warning: this constructor always performs a copy/move into a new, internally owned matrix.
        // If a const reference to an externally owned matrix is desired, use `Hessian::borrow` instead.
        template<typename T, std::enable_if_t<Types::template contains<std::decay_t<T>> && !std::is_same_v<std::decay_t<T>, Hessian>, int> = 0>
        Hessian(T &&H) : m_evaluated_hessian(std::in_place, std::in_place_type<std::decay_t<T>>, std::forward<T>(H)) { }

        template<typename T>
        static Hessian borrow(const T &hessian) {
            using HessianType = std::remove_cv_t<T>;
            static_assert(Types::template contains<HessianType>, "Unsupported Hessian representation");
            Hessian result;
            result.m_evaluated_hessian.emplace(std::in_place_type<std::reference_wrapper<const HessianType>>, std::cref(hessian));
            return result;
        }

        template <typename T, typename... Args>
        T &emplace(Args &&...args) {
            static_assert(Types::template contains<T>, "Unsupported Hessian representation");
            clear_caches();
            m_evaluated_hessian.emplace(std::in_place_type<T>, std::forward<Args>(args)...);
            return std::get<T>(*m_evaluated_hessian);
        }

        Hessian(const Hessian &other) = delete;            // Would require all Hessian representations to be copyable...
        Hessian(Hessian &&other) = default;
        Hessian &operator=(const Hessian &other) = delete; // Would require all Hessian representations to be copyable...
        Hessian &operator=(Hessian &&other) = default;

        template<typename T>
        static const T &unwrap(const T &H) { return H; }

        template<typename T>
        static const T &unwrap(const std::reference_wrapper<const T> &H) { return H.get(); }

        template<typename T>
        T &get_mutable() {
            if (!m_evaluated_hessian) throw std::logic_error("Hessian has not been evaluated");
            if (!std::holds_alternative<T>(*m_evaluated_hessian)) throw std::logic_error("Hessian is not of the requested non-reference type");
            if (has_cached_conversions()) throw std::logic_error("Cannot mutate Hessian after conversions have been cached; call clear_caches() first if you know this is safe (no dangling references to the cached conversions exist)");
            return std::get<T>(*m_evaluated_hessian);
        }

        // Replace m_evaluated_hessian with a new value of type T, clearing any
        // cached conversions.
        template<typename T>
        void switch_to_native_type() {
            if (std::holds_alternative<T>(m_value())) return; // Already owning the requested type

            if (const auto *H_ref = std::get_if<std::reference_wrapper<const T>>(&m_value())) {
                if constexpr (!std::is_copy_constructible_v<T>) throw std::logic_error("Cannot switch borrowed non-copyable Hessian to owned native type");
                else m_evaluated_hessian.emplace(std::in_place_type<T>, H_ref->get());
            }
            else m_evaluated_hessian.emplace(std::in_place_type<T>, std::move(const_cast<T &>(this->as<T>())));

            clear_caches();
        }

        template<typename T>
        const T &as() const {
            static_assert(Types::template contains<T>, "T is not one of the supported Hessian representations");

            if (const auto *H = std::get_if<T>(&m_value()))
                return *H;

            if (const auto *H = std::get_if<std::reference_wrapper<const T>>(&m_value()))
                return H->get();

            auto &cache = std::get<std::optional<T>>(m_conversion_caches);

            if (!cache) {
                cache.emplace(std::visit(
                    [](const auto &source) -> T { return HessianConversion::convert(HessianConversion::type_tag<T>{}, unwrap(source)); },
                    m_value()));
            }

            return *cache;
        }

        // Support "casting" to dense Hessian for duck typing with Eigen matrix types.
        // This should essentially never be used in practice (and hence is guarded by `explicit`),
        // however it is currently needed to build the unit tests.
        explicit operator const Eigen::MatrixXd &() const { return as<DenseHessian>(); }

        template <typename T>
        bool is_native_type() const noexcept {
            if (!m_evaluated_hessian) return false;
            static_assert(Types::template contains<T>, "T is not one of the supported Hessian representations");
            return std::holds_alternative<T>(m_value());
        }

        Eigen::VectorXd operator*(const Eigen::VectorXd &v) const { return std::visit([&v](const auto &H) -> Eigen::VectorXd { return unwrap(H) * v; }, m_value()); }
        Eigen::Index rows() const { return std::visit([](const auto &H) -> Eigen::Index { return unwrap(H).rows(); }, m_value()); }
        Eigen::Index cols() const { return std::visit([](const auto &H) -> Eigen::Index { return unwrap(H).cols(); }, m_value()); }

        // Empty state management
        bool has_value() const noexcept { return m_evaluated_hessian.has_value(); }
        explicit operator bool() const noexcept { return has_value(); }
        void reset() noexcept { m_evaluated_hessian.reset(); clear_caches(); }
    private:
        const Variant &m_value() const {
            if (!m_evaluated_hessian)
                throw std::logic_error("Hessian has not been evaluated");
            return *m_evaluated_hessian;
        }

        Variant &m_value() { return const_cast<Variant &>(static_cast<const Hessian &>(*this).m_value()); }
        void clear_caches() const { std::apply([](auto &...cache) { (cache.reset(), ...); }, m_conversion_caches); }
        bool has_cached_conversions() const { return std::apply([](const auto &...cache) { return (... || cache.has_value()); }, m_conversion_caches); }

        using Caches = typename Types::CacheTuple;

        std::optional<Variant> m_evaluated_hessian;
        mutable Caches m_conversion_caches; // Caches of converted Hessians.
    };
} // namespace polysolve
