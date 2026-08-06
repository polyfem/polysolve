#!/bin/bash
# Sweep the hybrid solver over rank counts on both MPI backends.
#
#   ./scripts/bench_hybrid.sh <openmpi-build> <thread-mpi-build> [grid] [reps]
#
# Ranks are processes in the first build and threads in the second; the driver
# (tests/bench_spmd.cpp) is the same source in both, which is the only way the
# comparison means anything.
set -u
MPI_BUILD=${1:?openmpi build dir}
TMPI_BUILD=${2:?thread-mpi build dir}
GRID=${3:-120}
REPS=${4:-3}
RANKS=${RANKS:-"1 2 4 8 16 32 64"}
OUT=${OUT:-bench_hybrid.csv}

# One sweep at a time, and never against a busy machine: both inflate results.
exec 9>"${TMPDIR:-/tmp}/.bench_hybrid.lock"
flock -n 9 || { echo "another sweep is running" >&2; exit 1; }

wait_quiet() {
    for _ in $(seq 60); do
        awk '{exit !($1 < 4.0)}' /proc/loadavg && return 0
        sleep 30
    done
    echo "machine never went quiet (load $(cut -d' ' -f1 /proc/loadavg)); refusing" >&2
    exit 1
}
wait_quiet

echo "backend,ranks,rep,grid,setup,solve,total,relres,loadavg" > "$OUT"

run() {   # backend ranks rep -> one csv row
    local backend=$1 n=$2 rep=$3 line load
    load=$(cut -d' ' -f1 /proc/loadavg)
    if [ "$backend" = openmpi ]; then
        line=$(mpirun -quiet --oversubscribe -np "$n" "$MPI_BUILD/tests/bench_spmd" "$GRID" 2>/dev/null | tail -1)
    else
        line=$(HYPRE_TMPI_NUM_THREADS=$n "$TMPI_BUILD/tests/bench_spmd" "$GRID" 2>/dev/null | tail -1)
    fi
    # bench_spmd prints: n=.. N=.. setup=.. solve=.. total=.. relres=..
    echo "$line" | awk -v b="$backend" -v r="$n" -v p="$rep" -v g="$GRID" -v l="$load" '
        /setup=/ {
            for (i = 1; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n", b, r, p, g, v["setup"], v["solve"], v["total"], v["relres"], l
        }'
}

# Interleave the backends inside each repetition so drift hits both equally.
for rep in $(seq "$REPS"); do
    for n in $RANKS; do
        run openmpi "$n" "$rep" >> "$OUT"
        run tmpi    "$n" "$rep" >> "$OUT"
        echo "rep $rep ranks $n done" >&2
    done
done
echo "wrote $OUT" >&2
