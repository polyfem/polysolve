#include "RankTeam.hpp"

#include <mpi.h>
#include <nanompi.h>

#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace polysolve::linear::mpi_utils
{
    namespace
    {
        struct Team
        {
            std::mutex mutex; // guards every field below
            std::condition_variable work_posted;
            std::condition_variable work_taken;

            nanompi_team *ranks = nullptr; // null when no team is running
            std::thread::id driver;
            int refs = 0;
            int size = 1;

            // The task being handed out; an empty one tells the workers to exit.
            std::function<void()> task;
            unsigned long generation = 0; // bumped for every task
            int taken = 0;                // workers that have copied the task
        };

        Team &team()
        {
            // Never destroyed: if a solver is leaked, its workers are still
            // waiting on these at exit, and destroying them then is undefined.
            static Team *t = new Team;
            return *t;
        }

        thread_local bool on_worker = false;

        // Body of ranks 1..n-1: take tasks until told to exit.
        int worker_main(void *)
        {
            on_worker = true;
            Team &t = team();
            unsigned long seen = 0;
            for (;;)
            {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(t.mutex);
                    t.work_posted.wait(lock, [&] { return t.generation != seen; });
                    seen = t.generation;
                    task = t.task;
                    if (++t.taken == t.size - 1)
                        t.work_taken.notify_one();
                }
                if (!task)
                    return 0;

                // The other ranks are waiting for this one inside the same MPI
                // calls, so a failure here cannot be recovered from.
                try
                {
                    task();
                }
                catch (const std::exception &e)
                {
                    std::fprintf(stderr, "polysolve: rank %d failed: %s\n", nanompi_rank(), e.what());
                    std::abort();
                }
                catch (...)
                {
                    std::fprintf(stderr, "polysolve: rank %d failed\n", nanompi_rank());
                    std::abort();
                }
            }
        }

        // Hands task to every worker and waits until each has taken a copy.
        // The caller holds t.mutex through lock.
        void post(Team &t, std::unique_lock<std::mutex> &lock, std::function<void()> task)
        {
            t.task = std::move(task);
            t.taken = 0;
            ++t.generation;
            t.work_posted.notify_all();
            t.work_taken.wait(lock, [&] { return t.taken == t.size - 1; });
            t.task = nullptr;
        }
    } // namespace

    bool RankTeam::acquire()
    {
        if (on_worker)
            return false;

        Team &t = team();
        std::lock_guard<std::mutex> lock(t.mutex);
        if (t.ranks)
        {
            if (t.driver != std::this_thread::get_id())
                return false;
            ++t.refs;
            return true;
        }

        const int n = nanompi_default_ranks();
        t.size = n < 1 ? 1 : n;
        t.task = nullptr;
        t.generation = 0;
        t.taken = 0;
        // The new workers block on t.mutex until this returns.
        if (nanompi_team_start(t.size, &worker_main, nullptr, &t.ranks) != 0)
        {
            t.ranks = nullptr;
            t.size = 1;
            throw std::runtime_error("polysolve: could not start the nano-mpi ranks");
        }
        t.driver = std::this_thread::get_id();
        t.refs = 1;
        return true;
    }

    void RankTeam::release()
    {
        Team &t = team();
        std::unique_lock<std::mutex> lock(t.mutex);
        assert(t.ranks && t.driver == std::this_thread::get_id() && t.refs > 0);
        if (!t.ranks || --t.refs > 0)
            return;

        post(t, lock, nullptr);
        // Joined with the lock held, so that no other thread can start a team
        // while this one's ranks still exist.
        nanompi_team_join(t.ranks);
        t.ranks = nullptr;
        t.driver = std::thread::id();
        t.size = 1;
    }

    void RankTeam::dispatch(std::function<void()> fn)
    {
        assert(fn);
        Team &t = team();
        std::unique_lock<std::mutex> lock(t.mutex);
        assert(t.ranks && t.driver == std::this_thread::get_id());
        if (t.size <= 1)
            return;
        post(t, lock, std::move(fn));
    }

    void RankTeam::run(const std::function<void()> &fn)
    {
        if (size() <= 1)
        {
            fn();
            return;
        }

        // The workers' copies refer to fn, so no rank may leave before all are done.
        dispatch([&fn] {
            fn();
            MPI_Barrier(MPI_COMM_WORLD);
        });
        fn();
        MPI_Barrier(MPI_COMM_WORLD);
    }

    bool RankTeam::is_worker()
    {
        return on_worker;
    }

    bool RankTeam::is_driver()
    {
        Team &t = team();
        std::lock_guard<std::mutex> lock(t.mutex);
        return t.ranks && t.driver == std::this_thread::get_id();
    }

    int RankTeam::size()
    {
        Team &t = team();
        std::lock_guard<std::mutex> lock(t.mutex);
        return t.ranks ? t.size : 1;
    }

} // namespace polysolve::linear::mpi_utils
