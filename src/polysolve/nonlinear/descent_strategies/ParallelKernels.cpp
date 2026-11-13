#include "ParallelKernels.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

#ifdef POLYSOLVE_WITH_TBB
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#endif

namespace polysolve::nonlinear::parallel
{
    int max_threads()
    {
#ifdef POLYSOLVE_WITH_TBB
        return int(tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism));
#else
        return 1;
#endif
    }

    int chunks(const Eigen::Index n, const Eigen::Index grain)
    {
        return int(std::max<Eigen::Index>(1, std::min<Eigen::Index>(max_threads(), n / std::max<Eigen::Index>(grain, 1))));
    }

    void for_chunks(const Eigen::Index n, const int c, const std::function<void(int, Eigen::Index, Eigen::Index)> &body)
    {
        if (c <= 1)
        {
            body(0, 0, n);
            return;
        }
#ifdef POLYSOLVE_WITH_TBB
        tbb::parallel_for(0, c, [&](const int i) { body(i, n * i / c, n * (i + 1) / c); });
#else
        for (int i = 0; i < c; ++i)
            body(i, n * i / c, n * (i + 1) / c);
#endif
    }

    void spmv_add(const StiffnessMatrix &H, const Eigen::VectorXd &q, const double alpha, Eigen::Ref<Eigen::VectorXd> y)
    {
        assert(q.size() == H.cols() && y.size() == H.rows());
        const auto *outer = H.outerIndexPtr();
        const auto *inner = H.innerIndexPtr();
        const auto *nnz = H.innerNonZeroPtr(); // null for a compressed matrix
        const double *val = H.valuePtr();
        const Eigen::Index n_cols = H.outerSize();
        for_chunks(H.rows(), chunks(H.rows(), 8192), [&](int, const Eigen::Index r0, const Eigen::Index r1) {
            for (Eigen::Index j = 0; j < n_cols; ++j)
            {
                const auto *e = inner + (nnz ? outer[j] + nnz[j] : outer[j + 1]);
                // the entries of column j in the rows [r0, r1)
                const auto *it = r0 == 0 ? inner + outer[j] : std::lower_bound(inner + outer[j], e, r0);
                const double qj = alpha * q[j];
                for (; it != e && *it < r1; ++it)
                    y[*it] += val[it - inner] * qj;
            }
        });
    }

    void spmv(const StiffnessMatrix &H, const Eigen::VectorXd &q, Eigen::Ref<Eigen::VectorXd> y)
    {
        y.setZero();
        spmv_add(H, q, 1.0, y);
    }

    void spadd(const StiffnessMatrix &A, const StiffnessMatrix &B, StiffnessMatrix &C)
    {
        if (!A.isCompressed() || !B.isCompressed() || A.rows() != B.rows() || A.cols() != B.cols())
        {
            C = A + B;
            C.makeCompressed();
            return;
        }
        using I = StiffnessMatrix::StorageIndex;
        const Eigen::Index n = A.outerSize();
        const I *ao = A.outerIndexPtr(), *ai = A.innerIndexPtr(), *bo = B.outerIndexPtr(), *bi = B.innerIndexPtr();
        const double *av = A.valuePtr(), *bv = B.valuePtr();
        const int c = chunks(n, 2048);

        // first pass: the size of each column of the union of the patterns
        std::vector<I> start(n + 1, 0);
        for_chunks(n, c, [&](int, const Eigen::Index j0, const Eigen::Index j1) noexcept {
            for (Eigen::Index j = j0; j < j1; ++j)
            {
                I a = ao[j], b = bo[j], k = 0;
                while (a < ao[j + 1] || b < bo[j + 1])
                {
                    if (a < ao[j + 1] && b < bo[j + 1] && ai[a] == bi[b])
                        ++a, ++b;
                    else if (a < ao[j + 1] && (b >= bo[j + 1] || ai[a] < bi[b]))
                        ++a;
                    else
                        ++b;
                    ++k;
                }
                start[j + 1] = k;
            }
        });
        for (Eigen::Index j = 0; j < n; ++j)
            start[j + 1] += start[j];

        C.resize(A.rows(), A.cols());
        C.resizeNonZeros(start[n]);
        std::copy(start.begin(), start.end(), C.outerIndexPtr());
        I *ci = C.innerIndexPtr();
        double *cv = C.valuePtr();
        // second pass: the entries of each column in index order, with the operations of Eigen's sum
        for_chunks(n, c, [&](int, const Eigen::Index j0, const Eigen::Index j1) noexcept {
            for (Eigen::Index j = j0; j < j1; ++j)
            {
                I a = ao[j], b = bo[j], k = start[j];
                while (a < ao[j + 1] || b < bo[j + 1])
                {
                    if (a < ao[j + 1] && b < bo[j + 1] && ai[a] == bi[b])
                    {
                        ci[k] = ai[a];
                        cv[k] = av[a] + bv[b];
                        ++a, ++b;
                    }
                    else if (a < ao[j + 1] && (b >= bo[j + 1] || ai[a] < bi[b]))
                    {
                        ci[k] = ai[a];
                        cv[k] = av[a] + 0.0;
                        ++a;
                    }
                    else
                    {
                        ci[k] = bi[b];
                        cv[k] = 0.0 + bv[b];
                        ++b;
                    }
                    ++k;
                }
            }
        });
    }
} // namespace polysolve::nonlinear::parallel
