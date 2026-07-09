#pragma once

#include <vector>

namespace polysolve::linear::hybrid
{
    class DisjointSet 
    {
    public:
        // Initializes a disjoint set of size n
        explicit DisjointSet(int n);

        // Finds the representative root of the set containing v (with path compression)
        int find_set(int v);

        // Unites the sets containing x and y (by rank)
        void union_set(int x, int y);

    private:
        std::vector<int> parent;
        std::vector<int> rank;
    };
} // namespace polysolve::linear::hybrid