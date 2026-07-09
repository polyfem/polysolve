#include "DisjointSet.hpp"

namespace polysolve::linear::hybrid
{
    DisjointSet::DisjointSet(int n) 
    {
        rank.assign(n, 0);
        parent.reserve(n);
        for (int i = 0; i < n; i++)
        {
            parent.push_back(i);
        }
    }

    int DisjointSet::find_set(int v) 
    {
        if (parent[v] != v)
        {
            parent[v] = find_set(parent[v]); // Path compression
        }
        return parent[v];
    }

    void DisjointSet::union_set(int x, int y) 
    {
        // Find the absolute root representatives first
        int root_x = find_set(x);
        int root_y = find_set(y);

        // If they already belong to the same set, do nothing
        if (root_x == root_y)
        {
            return;
        }

        // Union by rank optimization
        if (rank[root_x] > rank[root_y])
        {
            parent[root_y] = root_x;
        }
        else 
        {
            parent[root_x] = root_y;
            if (rank[root_x] == rank[root_y])
            {
                rank[root_y]++;
            }
        }
    }
} // namespace polysolve::linear::hybrid