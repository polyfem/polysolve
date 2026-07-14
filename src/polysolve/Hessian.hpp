// Support for various Hessian representations and conversions between them.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <MeshFEMSparse/BlockCSCHessian.hh>

#include <optional>
#include <variant>
#include <type_traits>

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
        std::vector<size_t> fixedVars;

        BCSCHessianWithFixedVars() = default;
        BCSCHessianWithFixedVars(std::unique_ptr<BCSCHessian> &&H_, const std::vector<size_t> &fixedVars_ = {})
            : H(std::move(H_)), fixedVars(fixedVars_) { }

        static BCSCHessianWithFixedVars fromEigen(const StiffnessMatrix &H_eigen, const std::vector<size_t> &fixedVars = {}) {
            return BCSCHessianWithFixedVars{BCSCHessian::fromEigen(H_eigen), fixedVars};
        }

        StiffnessMatrix toEigen() const { return H->template toEigen<StiffnessMatrix::StorageIndex>(/* upperTriangleOnly = */ false, fixedVars); }

        // Basic "duck typing" compatibility with Eigen matrix types.
        size_t rows() const { return H->numScalarCols(); }
        size_t cols() const { return H->numScalarCols(); }
        Eigen::VectorXd operator*(const Eigen::VectorXd &v) const {
            if (fixedVars.empty())  return H->apply(v);
            throw std::runtime_error("BCSCHessianWithFixedVars::operator* not yet implemented for nonempty fixedVars");
        }
    };

    struct HessianConversion {
        using DenseHessian = Eigen::MatrixXd;

        template <typename T>
        struct type_tag { using type = T; };

        template <typename Variant>
        struct VariantTraits;

        template <typename... Ts>
        struct VariantTraits<std::variant<Ts...>> {
            using CacheTuple = std::tuple<std::optional<Ts>...>;
            template <typename T>
            static constexpr bool contains = (std::is_same_v<T, Ts> || ...);
        };

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
        // static Eigen::MatrixXd convert(type_tag<Eigen::MatrixXd>, const StiffnessMatrix &H) { return Eigen::MatrixXd(H); } // To dense

        // TODO: this conversion can be avoided once MeshFEMSparse updates its
        // `CatamariFactorizer` wrapper to support Eigen sparse matrices natively.
        static BCSCHessianWithFixedVars convert(type_tag<BCSCHessianWithFixedVars>, const StiffnessMatrix &H) { return BCSCHessianWithFixedVars::fromEigen(H); }
        static StiffnessMatrix          convert(type_tag<StiffnessMatrix>, const BCSCHessianWithFixedVars &H) { return H.toEigen(); }
        static DenseHessian             convert(type_tag<DenseHessian>,    const BCSCHessianWithFixedVars &H) { return DenseHessian(H.toEigen()); } // TODO: Avoid double-conversion?
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
    struct Hessian {
        using DenseHessian = HessianConversion::DenseHessian;
        using Variant = std::variant<StiffnessMatrix, DenseHessian, BCSCHessianWithFixedVars>;

        Hessian() = default;

        template<typename T, std::enable_if_t<std::is_constructible_v<Variant, T &&> && !std::is_same_v<std::decay_t<T>, Hessian>, int> = 0>
        Hessian(T &&H) : m_evaluated_hessian(std::in_place, std::in_place_type<std::decay_t<T>>, std::forward<T>(H)) { }

        template <typename T, typename... Args>
        T &emplace(Args &&...args) {
            static_assert(Traits::template contains<T>, "Unsupported Hessian representation");
            clear_caches();
            m_evaluated_hessian.emplace(std::in_place_type<T>, std::forward<Args>(args)...);
            return std::get<T>(*m_evaluated_hessian);
        }

        Hessian &operator=(const Hessian &other) = delete; // Would require all Hessian representations to be copyable...
        Hessian &operator=(Hessian &&other) = default;

        template<typename T>
        T &get_mutable() {
            if (!m_evaluated_hessian) throw std::logic_error("Hessian has not been evaluated");
            if (!std::holds_alternative<T>(*m_evaluated_hessian)) throw std::logic_error("Hessian is not of the requested type");
            if (has_cached_conversions()) throw std::logic_error("Cannot mutate Hessian after conversions have been cached; call clear_caches() first if you know this is safe (no dangling references to the cached conversions exist)");
            return std::get<T>(*m_evaluated_hessian);
        }

        // Replace m_evaluated_hessian with a new value of type T, clearing any
        // cached conversions.
        template<typename T>
        void switch_to_native_type() {
            if (is_native_type<T>()) return; // Already in the requested type
            m_evaluated_hessian.emplace(std::in_place_type<T>, std::move(const_cast<T &>(this->as<T>())));
            clear_caches();
        }

        template<typename T>
        const T &as() const {
            static_assert(Traits::template contains<T>, "T is not one of the supported Hessian representations");

            if (const auto *H = std::get_if<T>(&m_value()))
                return *H;

            auto &cache = std::get<std::optional<T>>(m_conversion_caches);

            if (!cache) {
                cache.emplace(std::visit(
                    [](const auto &source) -> T { return HessianConversion::convert(HessianConversion::type_tag<T>{}, source); },
                    m_value()));
            }

            return *cache;
        }

        template <typename T>
        bool is_native_type() const noexcept {
            if (!m_evaluated_hessian) return false;
            static_assert(Traits::template contains<T>, "T is not one of the supported Hessian representations");
            return std::holds_alternative<T>(m_value());
        }

        Eigen::VectorXd operator*(const Eigen::VectorXd &v) const { return std::visit([&v](const auto &H) -> Eigen::VectorXd { return H * v; }, m_value()); }
        Eigen::Index rows() const { return std::visit([](const auto &H) -> Eigen::Index { return H.rows(); }, m_value()); }
        Eigen::Index cols() const { return std::visit([](const auto &H) -> Eigen::Index { return H.cols(); }, m_value()); }

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

        using Traits = HessianConversion::VariantTraits<Variant>;
        using Caches = typename Traits::CacheTuple;

        std::optional<Variant> m_evaluated_hessian;
        mutable Caches m_conversion_caches; // Caches of converted Hessians.
    };
} // namespace polysolve
