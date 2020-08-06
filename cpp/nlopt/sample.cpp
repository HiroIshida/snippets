#include <nlopt.hpp>
#include <iostream>

double myfunc(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data);

void multi_constraint(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data);

int main()
{
	nlopt::opt opt(nlopt::LD_SLSQP, 2);
	std::vector<double> lb(2);
	lb[0] = -HUGE_VAL;   //HUGE_VAL is a C++ constant
	lb[1] = 0;
	opt.set_lower_bounds(lb);

	opt.set_min_objective(myfunc, NULL);

	double data[4] = {2,0,-1,1};   //use one dimensional array
	std::vector<double> tol_constraint(2);
	tol_constraint[0] = 1e-8;
	tol_constraint[1] = 1e-8;
	opt.add_inequality_mconstraint(multi_constraint, data, tol_constraint);
	opt.set_xtol_rel(1e-4);

	std::vector<double> x(2);
	x[0] = 1.234;
	x[1] = 5.678;
	double minf;
	nlopt::result result = opt.optimize(x, minf);
	std::cout << "The result is" << std::endl;
	std::cout << result << std::endl;
	std::cout << "Minimal function value " << minf << std::endl;
}

double myfunc(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
    if (!grad.empty()) {
        grad[0] = 0.0;
        grad[1] = 0.5 / sqrt(x[1]);
    }
    return sqrt(x[1]);
}

void multi_constraint(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data)
{
    //n is the length of x, m is the length of result
    double *df_data = static_cast<double*>(f_data);
    double a1 = df_data[0];
    double b1 = df_data[1];
    double a2 = df_data[2];
    double b2 = df_data[3];

    //The n dimension of grad is stored contiguously, so that \partci/\partxj is stored in grad[i*n + j]
    //Here you see take dCi/dx0...dxn and store it one by one, then repeat. grad is just an one dimensional array

    if (grad) {
        grad[0] = 3 * a1 * (a1*x[0] + b1) * (a1*x[0] + b1);
        grad[1] = -1.0;
        grad[2] = 3 * a2 * (a2*x[0] + b2) * (a2*x[0] + b2);
        grad[3] = -1.0;
    }

    result[0] = ((a1*x[0] + b1) * (a1*x[0] + b1) * (a1*x[0] + b1) - x[1]);
    result[1] = ((a2*x[0] + b2) * (a2*x[0] + b2) * (a2*x[0] + b2) - x[1]);

    return;
}
