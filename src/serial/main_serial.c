#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "common/gauss_rule.h"
#include "common/integrands.h"

double integrate_1d_composite(
    Integrand1DFn func,
    double a,
    double b,
    long long num_panels,
    GaussRule* rule
);

int main (int argc, char** argv) {
    GaussRule gauss_rule;
    gauss_rule_init(&gauss_rule, 8);

    const Integrand1D* integrand = get_integrand_1d(FUNC_POLY);
    double res = integrate_1d_composite(
        integrand->integrand_fn,
        integrand->default_a,
        integrand->default_b,
        1,
        &gauss_rule
    );

    printf("result: %.15f (exact: %.15f, err: %.3e)\n",
           res, integrand->exact_value, fabs(res - integrand->exact_value));

    gauss_rule_free(&gauss_rule);
    return 0;
}

double integrate_1d_composite(
    Integrand1DFn func,
    double a,
    double b,
    long long num_panels,
    GaussRule* rule
) {
    double h = (b-a)/(double)num_panels;    // width of a panel
    double half_h = 0.5 * h;                // half width of the panel
    double total = 0.0;                     // the final result
    double remainder = 0.0;                 // kahan remainder

    for (long long i = 0; i < num_panels; ++i) {
        double left = a + i * h;
        double mid = left + half_h;
        double panel_sum = 0.0;
        for (int k = 0; k < rule->order; ++k) {
            double x = mid + half_h * rule->nodes[k];
            panel_sum += rule->weights[k] * func(x);
        }

        double panel_area = half_h * panel_sum;
        double y = panel_area - remainder;
        double t = total + y;
        remainder = (t - total) - y;
        total = t;
    }

    return total;
}
