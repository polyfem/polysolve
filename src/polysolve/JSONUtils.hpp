#pragma once

#include "Types.hpp"


namespace nlohmann
{
	template <typename T, int nrows, int ncols, int options, int maxdim1, int maxdim2>
	struct adl_serializer<Eigen::Matrix<T, nrows, ncols, options, maxdim1, maxdim2>>
	{
		static void to_json(json &j, const Eigen::Matrix<T, nrows, ncols, options, maxdim1, maxdim2> &matrix)
		{
			if (matrix.rows() > 1 && matrix.cols() > 1)
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
			else
			{
				j = json::array();
				for (int i = 0; i < matrix.size(); ++i)
				{
					j.push_back(matrix(i));
				}
			}
		}

		static void from_json(const json &j, Eigen::Matrix<T, nrows, ncols, options, maxdim1, maxdim2> &matrix)
		{
			using Scalar = typename Eigen::Matrix<T, nrows, ncols, options, maxdim1, maxdim2>::Scalar;

			if (j.is_number())
			{
				matrix.resize((nrows == -1) ? 1 : nrows, (ncols == -1) ? 1 : ncols);
				matrix(0, 0) = j.get<Scalar>();
			}
			else if (j.is_array())
			{
				if (j.size() == 0) // empty array
					matrix.setZero(std::max(0, nrows), std::max(0, ncols));
				else if (j.at(0).is_number()) // 1D array
				{
					assert(nrows == -1 || ncols == -1); // at least one dimension can be resized
					if (ncols == -1)
						matrix.setZero(1, j.size());
					else
						matrix.setZero(j.size(), 1);
					
					for (int i = 0; i < j.size(); i++)
						matrix(i) = j.at(i).get<Scalar>();
				}
				else // 2D array
				{
					matrix.setZero(j.size(), j.at(0).size());
					
					for (int r = 0; r < matrix.rows(); r++)
					{
						const auto& jrow = j.at(r);
						for (int c = 0; c < matrix.cols(); c++)
						{
							matrix(r, c) = jrow.at(c).get<Scalar>();
						}
					}
				}
			}
			else
				assert(false);
		}
	};
} // namespace nlohmann
