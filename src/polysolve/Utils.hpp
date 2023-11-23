#pragma once

#include "Types.hpp"

#include <spdlog/spdlog.h>

#define POLYSOLVE_SCOPED_STOPWATCH(...) polysolve::StopWatch __polysolve_stopwatch(__VA_ARGS__)

namespace polysolve
{

    struct Timing
    {
        operator double() const { return time; }

        void operator+=(const double t)
        {
            time += t;
            ++count;
        }

        double time = 0;
        size_t count = 0;
    };

    class StopWatch
    {
    private:
        using clock = std::chrono::steady_clock;

    public:
        StopWatch(const std::string &name, spdlog::logger &logger);
        StopWatch(const std::string &name, double &total_time, spdlog::logger &logger);

        virtual ~StopWatch();

        void start();
        void stop();

        double getElapsedTimeInSec();
        void log_msg();

    private:
        std::string m_name;
        std::chrono::time_point<clock> m_start, m_stop;
        double *m_total_time = nullptr;
        size_t *m_count = nullptr;
        bool is_running = false;

        spdlog::logger &m_logger;
    };

    [[noreturn]] void log_and_throw_error(spdlog::logger &logger, const std::string &msg);

    template <typename... Args>
    [[noreturn]] void log_and_throw_error(spdlog::logger &logger, const std::string &msg, const Args &...args)
    {
        log_and_throw_error(logger, fmt::format(msg, args...));
    }

    Eigen::SparseMatrix<double> sparse_identity(int rows, int cols);

} // namespace polysolve

namespace nlohmann
{
	template <typename T, int nrows, int ncols, int maxdim1, int maxdim2>
	struct adl_serializer<Eigen::Matrix<T, nrows, ncols, Eigen::ColMajor, maxdim1, maxdim2>>
	{
		static void to_json(json &j, const Eigen::Matrix<T, nrows, ncols, Eigen::ColMajor, maxdim1, maxdim2> &matrix)
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