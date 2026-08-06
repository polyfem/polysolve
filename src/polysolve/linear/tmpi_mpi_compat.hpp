#pragma once
//
// The few MPI facilities hypre's thread-MPI backend does not already provide.
//
// With HYPRE_SEQUENTIAL (which the thread-MPI backend builds under), hypre's
// mpistubs.h already rewrites MPI_Bcast, MPI_Allreduce, MPI_Barrier,
// MPI_Scatterv, MPI_Gatherv, MPI_COMM_WORLD, MPI_INT, MPI_DOUBLE, MPI_SUM and
// friends onto hypre_MPI_*, so those calls need no help.
//
// Three things are missing, and this header supplies them:
//
//   * MPI-3 shared-memory windows. They exist so ranks in separate address
//     spaces can share a buffer; when the ranks are threads they already do,
//     so a window is just "one rank allocates, everybody learns the pointer".
//   * MPI_IN_PLACE, which hypre's reduction layer has no notion of.
//   * MPI_Init / MPI_Initialized / MPI_Finalized, which the ranks do not need
//     because hypre_tmpi_team_create() has already made them exist.
//
#include <HYPRE_utilities.h>
#include <_hypre_utilities.h>
#include "mpithreads.h"

#include <cstdlib>
#include <cstring>

namespace polysolve::tmpi_compat
{
    inline size_t datatype_size(hypre_MPI_Datatype dt)
    {
        switch (dt)
        {
        case hypre_MPI_INT:            return sizeof(HYPRE_Int);
        case hypre_MPI_LONG_LONG_INT:  return sizeof(HYPRE_BigInt);
        case hypre_MPI_DOUBLE:         return sizeof(double);
        case hypre_MPI_REAL:           return sizeof(HYPRE_Real);
        case hypre_MPI_FLOAT:          return sizeof(float);
        case hypre_MPI_CHAR:           return sizeof(char);
        case hypre_MPI_BYTE:           return 1;
        default:                       return 0;
        }
    }
} // namespace polysolve::tmpi_compat

// ---------------------------------------------------------------- IN_PLACE --
#ifndef MPI_IN_PLACE
#define MPI_IN_PLACE ((void *) -1L)
#endif

// hypre_MPI_Allreduce has no in-place path, so stage the input through a copy.
#undef MPI_Allreduce
static inline HYPRE_Int MPI_Allreduce(const void *sendbuf, void *recvbuf, HYPRE_Int count,
                                      hypre_MPI_Datatype dt, hypre_MPI_Op op,
                                      hypre_MPI_Comm comm)
{
    if (sendbuf == MPI_IN_PLACE)
    {
        const size_t nb = polysolve::tmpi_compat::datatype_size(dt) * (size_t) count;
        void *tmp = std::malloc(nb ? nb : 1);
        std::memcpy(tmp, recvbuf, nb);
        const HYPRE_Int rc = hypre_MPI_Allreduce(tmp, recvbuf, count, dt, op, comm);
        std::free(tmp);
        return rc;
    }
    return hypre_MPI_Allreduce(const_cast<void *>(sendbuf), recvbuf, count, dt, op, comm);
}

// ------------------------------------------------- const-correct collectives --
// The MPI standard gives send buffers as `const void *`; hypre's equivalents
// take `void *`. Wrap the few polysolve uses so const data still compiles.
#undef MPI_Scatterv
static inline HYPRE_Int MPI_Scatterv(const void *sendbuf, HYPRE_Int *sendcounts,
                                     HYPRE_Int *displs, hypre_MPI_Datatype sendtype,
                                     void *recvbuf, HYPRE_Int recvcount,
                                     hypre_MPI_Datatype recvtype, HYPRE_Int root,
                                     hypre_MPI_Comm comm)
{
    return hypre_MPI_Scatterv(const_cast<void *>(sendbuf), sendcounts, displs, sendtype,
                              recvbuf, recvcount, recvtype, root, comm);
}

#undef MPI_Gatherv
static inline HYPRE_Int MPI_Gatherv(const void *sendbuf, HYPRE_Int sendcount,
                                    hypre_MPI_Datatype sendtype, void *recvbuf,
                                    HYPRE_Int *recvcounts, HYPRE_Int *displs,
                                    hypre_MPI_Datatype recvtype, HYPRE_Int root,
                                    hypre_MPI_Comm comm)
{
    return hypre_MPI_Gatherv(const_cast<void *>(sendbuf), sendcount, sendtype, recvbuf,
                             recvcounts, displs, recvtype, root, comm);
}

#undef MPI_Bcast
static inline HYPRE_Int MPI_Bcast(void *buffer, HYPRE_Int count, hypre_MPI_Datatype dt,
                                  HYPRE_Int root, hypre_MPI_Comm comm)
{
    return hypre_MPI_Bcast(buffer, count, dt, root, comm);
}

// ----------------------------------------------------------------- windows --
struct tmpi_win_s
{
    void          *base    = nullptr;
    size_t         bytes   = 0;
    int            disp    = 1;
    int            i_alloc = 0;      // this rank owns the allocation
    hypre_MPI_Comm comm    = hypre_MPI_COMM_WORLD;
};
typedef tmpi_win_s *MPI_Win;

#ifndef MPI_MODE_NOPRECEDE
#define MPI_MODE_NOPRECEDE 0
#define MPI_MODE_NOSUCCEED 0
#endif

// One rank passes a non-zero size and allocates; the pointer is then broadcast,
// which is the whole of "shared memory" when the ranks are threads.
static inline HYPRE_Int MPI_Win_allocate_shared(hypre_MPI_Aint size, int disp_unit,
                                                hypre_MPI_Info info, hypre_MPI_Comm comm,
                                                void *baseptr, MPI_Win *win)
{
    (void) info;
    HYPRE_Int me = 0;
    hypre_MPI_Comm_rank(comm, &me);

    tmpi_win_s *w = new tmpi_win_s();
    w->comm = comm;
    w->disp = disp_unit;

    void *p = nullptr;
    if (size > 0)
    {
        p = std::malloc((size_t) size);
        std::memset(p, 0, (size_t) size);
        w->bytes   = (size_t) size;
        w->i_alloc = 1;
    }

    // whoever allocated is rank 0 in every polysolve use; publish its pointer
    hypre_MPI_Bcast(&p, (HYPRE_Int) sizeof(void *), hypre_MPI_BYTE, 0, comm);
    hypre_MPI_Bcast(&w->bytes, (HYPRE_Int) sizeof(size_t), hypre_MPI_BYTE, 0, comm);

    w->base = p;
    *(void **) baseptr = p;
    *win = w;
    return 0;
}

static inline HYPRE_Int MPI_Win_shared_query(MPI_Win win, int rank, hypre_MPI_Aint *size,
                                             int *disp_unit, void *baseptr)
{
    (void) rank;
    if (size)      { *size = (hypre_MPI_Aint) win->bytes; }
    if (disp_unit) { *disp_unit = win->disp; }
    *(void **) baseptr = win->base;
    return 0;
}

// Threads share the buffer outright, so a fence is only a synchronisation point.
static inline HYPRE_Int MPI_Win_fence(int assert_, MPI_Win win)
{
    (void) assert_;
    return hypre_MPI_Barrier(win->comm);
}

static inline HYPRE_Int MPI_Win_free(MPI_Win *win)
{
    if (!win || !*win) { return 0; }
    tmpi_win_s *w = *win;
    hypre_MPI_Barrier(w->comm);          // nobody may still be reading it
    if (w->i_alloc) { std::free(w->base); }
    delete w;
    *win = nullptr;
    return 0;
}

// ------------------------------------------------------------ init/finalize --
// The team already created the ranks, so these only report the state.
#undef MPI_Init
#undef MPI_Initialized
#undef MPI_Finalized
static inline HYPRE_Int MPI_Init(int *, char ***)      { return 0; }
static inline HYPRE_Int MPI_Initialized(int *flag)     { *flag = 1; return 0; }
static inline HYPRE_Int MPI_Finalized(int *flag)       { *flag = 0; return 0; }
