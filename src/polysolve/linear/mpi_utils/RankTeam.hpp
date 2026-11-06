#pragma once

#include <functional>

namespace polysolve::linear::mpi_utils
{
    // The nano-mpi ranks shared by every solver in the process.
    //
    // MPI here is nano-mpi, whose ranks are threads of this process, and it
    // allows one team of ranks per process. So solvers cannot each start their
    // own: they share this one. The first acquire() starts it, making the
    // calling thread rank 0 (the "driver") and spawning ranks 1..n-1 as worker
    // threads. The last release() stops and joins them.
    //
    // Workers sleep on a condition variable between tasks. Waiting inside an
    // MPI call instead would spin, since nano-mpi's collectives busy-wait, and
    // a solver can sit idle for a long time between solves.
    //
    // The number of ranks is nano-mpi's default: NANOMPI_NUM_RANKS if set,
    // otherwise the number of cores.
    class RankTeam
    {
    public:
        // Takes a reference on the team, starting it if needed. Returns false,
        // and takes no reference, when the calling thread cannot drive the
        // team: it is itself a worker rank, or another thread started the team
        // (nano-mpi ties a rank to the thread that plays it).
        static bool acquire();

        // Drops a reference taken by acquire(). The last one stops the team.
        static void release();

        // Driver only. Hands a copy of fn to every worker and returns once each
        // has taken it, so that the driver can then run its own part of the
        // same SPMD code. fn must take part in the same MPI calls as the driver.
        static void dispatch(std::function<void()> fn);

        // Driver only. Runs fn on every rank, the driver included, and returns
        // once all of them have finished. fn may use MPI_COMM_WORLD.
        static void run(const std::function<void()> &fn);

        // True on ranks 1..n-1.
        static bool is_worker();

        // True on the thread that started the live team (rank 0).
        static bool is_driver();

        // Number of ranks in the live team, or 1 when there is none.
        static int size();
    };

} // namespace polysolve::linear::mpi_utils
