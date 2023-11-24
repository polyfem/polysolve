#pragma once

#include "Types.hpp"

namespace nlohmann
{
	template <typename T, int nrows, int ncols, int maxdim1, int maxdim2>
	struct adl_serializer<Eigen::Matrix<T, nrows, ncols, Eigen::ColMajor, maxdim1, maxdim2>>
	{
		static void to_json(json &j, const Eigen::Matrix<T, nrows, ncols, Eigen::ColMajor, maxdim1, maxdim2> &matrix)
		{
			if (matrix.rows() > 1)
			{
				for (int row = 0; row < matrix.rows(); ++row)
				{
					json column = json::array();
					for (int col = 0; col < matrix.cols(); ++col)
					{
						column.push_back(matrix(row, col));
					}
					j.push_back(column);
				}
			}
			else if (matrix.rows() == 1)
			{
				j = json::array();
				for (int col = 0; col < matrix.cols(); ++col)
				{
					j.push_back(matrix(0, col));
				}
			}
		}

		static void from_json(const json &j, Eigen::Matrix<T, nrows, ncols, Eigen::ColMajor, maxdim1, maxdim2> &matrix)
		{
			using Scalar = typename Eigen::Matrix<T, nrows, ncols, Eigen::ColMajor, maxdim1, maxdim2>::Scalar;
			assert(j.size() > 0);
			assert(nrows >= j.size() || nrows == -1);
			assert(j.at(0).is_number() || ncols == -1 || ncols >= j.at(0).size());
			
			const int n_cols = j.at(0).is_number() ? 1 : j.at(0).size();
			if (nrows == -1 || ncols == -1)
				matrix.setZero(j.size(), n_cols);
			
			for (std::size_t row = 0; row < j.size(); ++row)
			{
				const auto& jrow = j.at(row);
				if (jrow.is_number())
					matrix(row) = jrow;
				else
					for (std::size_t col = 0; col < jrow.size(); ++col)
					{
						const auto& value = jrow.at(col);
						matrix(row, col) = value.get<Scalar>();
					}
			}
		}
	};
} // namespace nlohmann
