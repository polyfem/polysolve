#include "LineSearch.hpp"

#include "Armijo.hpp"
#include "Backtracking.hpp"
#include "CppOptArmijo.hpp"
#include "MoreThuente.hpp"

#include <polysolve/nonlinear/Logger.hpp>

#include <cfenv>
#include <fstream>

namespace polysolve::nonlinear::line_search
{
    std::shared_ptr<LineSearch> LineSearch::construct_line_search(const std::string &name, const std::shared_ptr<Logger> &logger)
    {
        if (name == "armijo" || name == "Armijo")
        {
            return std::make_shared<Armijo>(logger);
        }
        else if (name == "armijo_alt" || name == "ArmijoAlt")
        {
            return std::make_shared<CppOptArmijo>(logger);
        }
        else if (name == "bisection" || name == "Bisection")
        {
            logger->warn("{} linesearch was renamed to \"backtracking\"; using backtracking line-search", name);
            return std::make_shared<Backtracking>(logger);
        }
        else if (name == "backtracking" || name == "Backtracking")
        {
            return std::make_shared<Backtracking>(logger);
        }
        else if (name == "more_thuente" || name == "MoreThuente")
        {
            return std::make_shared<MoreThuente>(logger);
        }
        else if (name == "none")
        {
            return nullptr;
        }
        else
        {
            logger->log_and_throw_error("Unknown line search {}!", name);
            return nullptr;
        }
    }

    void LineSearch::save_sampled_values(
        const std::string &filename,
        const typename Problem::TVector &x,
        const typename Problem::TVector &delta_x,
        Problem &objFunc,
        const Logger &logger,
        const double starting_step_size,
        const int num_samples)
    {
        std::ofstream samples(filename, std::ios::out);
        if (!samples.is_open())
        {
            logger.log_and_throw_error("Unable to save sampled values to file \"{}\" !", filename);
        }

        samples << "alpha,f(x + alpha * delta_x),valid,decrease\n";

        objFunc.solution_changed(x);
        double fx = objFunc.value(x);

        Eigen::VectorXd alphas = Eigen::VectorXd::LinSpaced(2 * num_samples - 1, -starting_step_size, starting_step_size);
        for (int i = 0; i < alphas.size(); i++)
        {
            Problem::TVector new_x = x + alphas[i] * delta_x;
            objFunc.solution_changed(new_x);
            double fxi = objFunc.value(new_x);
            samples << alphas[i] << ","
                    << fxi << ","
                    << (objFunc.is_step_valid(x, new_x) ? "true" : "false") << ","
                    << (fxi < fx ? "true" : "false") << "\n";
        }

        samples.close();
    }

    LineSearch::LineSearch(const std::shared_ptr<Logger> &logger)
        : m_logger(logger)
    {
    }

    double LineSearch::compute_nan_free_step_size(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const double starting_step_size,
        const double rate)
    {
        double step_size = starting_step_size;
        TVector new_x = x + step_size * delta_x;

        // Find step that does not result in nan or infinite energy
        while (step_size > min_step_size && cur_iter < max_step_size_iter)
        {
            // Compute the new energy value without contacts
            // TODO: removed only elastic
            const double energy = objFunc.value(new_x);
            const bool is_step_valid = objFunc.is_step_valid(x, new_x);

            if (!std::isfinite(energy) || !is_step_valid)
            {
                step_size *= rate;
                new_x = x + step_size * delta_x;
            }
            else
            {
                break;
            }
            cur_iter++;
        }

        if (cur_iter >= max_step_size_iter || step_size <= min_step_size)
        {
            m_logger->error(
                "Line search failed to find a valid finite energy step (cur_iter={:d} step_size={:g})!",
                cur_iter, step_size);
            return std::nan("");
        }

        return step_size;
    }

    double LineSearch::compute_collision_free_step_size(
        const TVector &x,
        const TVector &delta_x,
        Problem &objFunc,
        const double starting_step_size)
    {
        double step_size = starting_step_size;
        TVector new_x = x + step_size * delta_x;

        // Find step that is collision free
        double max_step_size = objFunc.max_step_size(x, new_x);
        if (max_step_size == 0)
        {
            m_logger->error("Line search failed because CCD produced a stepsize of zero!");
            objFunc.line_search_end();
            return std::nan("");
        }

        { // clang-format off
				//#pragma STDC FENV_ACCESS ON
				const int current_round = std::fegetround();
				std::fesetround(FE_DOWNWARD);
				step_size *= max_step_size; // TODO: check me if correct
				std::fesetround(current_round);
				} // clang-format on

        // m_logger->trace("\t\tpre TOI={}, ss={}", max_step_size, step_size);

        // while (max_step_size != 1)
        // {
        // 	new_x = x + step_size * delta_x;
        // 	max_step_size = objFunc.max_step_size(x, new_x);
        //
        // 	std::fesetround(FE_DOWNWARD);
        // 	step_size *= max_step_size; // TODO: check me if correct
        // 	std::fesetround(current_roudn);
        // 	if (max_step_size != 1)
        // 		m_logger->trace("\t\trepeating TOI={}, ss={}", max_step_size, step_size);
        // }

        return step_size;
    }

    // #ifndef NDEBUG
    // 			template <typename ProblemType>
    // 			double LineSearch::compute_debug_collision_free_step_size(
    // 				const TVector &x,
    // 				const TVector &delta_x,
    // 				ProblemType &objFunc,
    // 				const double starting_step_size,
    // 				const double rate)
    // 			{
    // 				double step_size = starting_step_size;

    // 				TVector new_x = x + step_size * delta_x;
    // 				{
    // 					POLYSOLVE_SCOPED_TIMER("constraint set update in LS", this->constraint_set_update_time);
    // 					objFunc.solution_changed(new_x);
    // 				}

    // 				// safe guard check
    // 				while (!objFunc.is_step_collision_free(x, new_x))
    // 				{
    // 					m_logger->error("step is not collision free!!");
    // 					step_size *= rate;
    // 					new_x = x + step_size * delta_x;
    // 					{
    // 						POLYSOLVE_SCOPED_TIMER("constraint set update in LS", this->constraint_set_update_time);
    // 						objFunc.solution_changed(new_x);
    // 					}
    // 				}
    // 				assert(objFunc.is_step_collision_free(x, new_x));

    // 				return step_size;
    // 			}
    // #endif
} // namespace polysolve::nonlinear::line_search
