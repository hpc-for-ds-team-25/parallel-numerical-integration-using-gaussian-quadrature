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

double integrate_3d_composite(
    Integrand3DFn func,
    double a,
    double b,
    long long num_panels,
    const GaussRule* rule
);

double integrate_1d_adaptive(
    Integrand1DFn func,
    double a,
    double b,
    long long initial_panels,
    double tol,
    int max_depth,
    const GaussRule* rule
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

    printf("1D (%s): %.15f (exact: %.15f, err: %.3e)\n",
           integrand->name, res, integrand->exact_value,
           fabs(res - integrand->exact_value));

    const Integrand3D* integrand_3d = get_integrand_3d(FUNC_HEAVY_3D);
    double res_3d = integrate_3d_composite(
        integrand_3d->integrand_fn,
        integrand_3d->default_a,
        integrand_3d->default_b,
        4,
        &gauss_rule
    );

    printf("3D (%s): %.15f (exact: %.15f, err: %.3e)\n",
           integrand_3d->name, res_3d, integrand_3d->exact_value,
           fabs(res_3d - integrand_3d->exact_value));

    const Integrand1D* peak = get_integrand_1d(FUNC_PEAK_1D);
    double res_adapt = integrate_1d_adaptive(
        peak->integrand_fn,
        peak->default_a,
        peak->default_b,
        1000,
        1e-12,
        30,
        &gauss_rule
    );

    printf("Adaptive (%s): %.15f (exact: %.15f, err: %.3e)\n",
           peak->name, res_adapt, peak->exact_value,
           fabs(res_adapt - peak->exact_value));

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

double integrate_3d_composite(
    Integrand3DFn func,
    double a,
    double b,
    long long num_panels,
    const GaussRule* rule
) {
    double h = (b - a) / (double)num_panels;
    double half_h = 0.5 * h;
    double cell_scale = half_h * half_h * half_h;
    double total = 0.0;
    double remainder = 0.0;

    for (long long kx = 0; kx < num_panels; ++kx) {
        double mx = a + kx * h + half_h;
        for (long long ky = 0; ky < num_panels; ++ky) {
            double my = a + ky * h + half_h;
            for (long long kz = 0; kz < num_panels; ++kz) {
                double mz = a + kz * h + half_h;
                double cell_sum = 0.0;

                for (int ix = 0; ix < rule->order; ++ix) {
                    double x = mx + half_h * rule->nodes[ix];
                    double wx = rule->weights[ix];
                    for (int iy = 0; iy < rule->order; ++iy) {
                        double y = my + half_h * rule->nodes[iy];
                        double wxy = wx * rule->weights[iy];
                        for (int iz = 0; iz < rule->order; ++iz) {
                            double z = mz + half_h * rule->nodes[iz];
                            cell_sum += wxy * rule->weights[iz] * func(x, y, z);
                        }
                    }
                }

                double cell_vol = cell_scale * cell_sum;
                double cy = cell_vol - remainder;
                double ct = total + cy;
                remainder = (ct - total) - cy;
                total = ct;
            }
        }
    }

    return total;
}

typedef struct {
    double l;
    double r;
    double coarse_val;
    int depth;
} AdaptiveTask;

static double eval_panel_1d(Integrand1DFn func, double l, double r, const GaussRule* rule) {
    double mid = 0.5 * (l + r);
    double half_h = 0.5 * (r - l);
    double sum = 0.0;
    for (int i = 0; i < rule->order; ++i) {
        double x = mid + half_h * rule->nodes[i];
        sum += rule->weights[i] * func(x);
    }
    return half_h * sum;
}

double integrate_1d_adaptive(
    Integrand1DFn func,
    double a,
    double b,
    long long initial_panels,
    double tol,
    int max_depth,
    const GaussRule* rule
) {
    if (initial_panels < 1) {
        initial_panels = 1;
    }

    size_t capacity = (size_t)initial_panels + 64;
    AdaptiveTask* stack = malloc(capacity * sizeof(AdaptiveTask));
    if (stack == NULL) {
        fprintf(stderr, "Error: failed to allocate adaptive stack\n");
        exit(EXIT_FAILURE);
    }

    size_t top = 0;
    double h = (b - a) / (double)initial_panels;
    for (long long k = initial_panels - 1; k >= 0; --k) {
        double l = a + k * h;
        double r = (k == initial_panels - 1) ? b : (a + (k + 1) * h);
        stack[top++] = (AdaptiveTask){l, r, eval_panel_1d(func, l, r, rule), 0};
    }

    double total = 0.0;
    double remainder = 0.0;
    double inv_domain = 1.0 / (b - a);

    while (top > 0) {
        AdaptiveTask task = stack[--top];
        double m = 0.5 * (task.l + task.r);
        double left_val = eval_panel_1d(func, task.l, m, rule);
        double right_val = eval_panel_1d(func, m, task.r, rule);
        double fine_val = left_val + right_val;

        double err_est = fabs(fine_val - task.coarse_val);
        double local_tol = tol * (task.r - task.l) * inv_domain;

        if (err_est <= local_tol || task.depth >= max_depth) {
            double y = fine_val - remainder;
            double t = total + y;
            remainder = (t - total) - y;
            total = t;
        } else {
            if (top + 2 > capacity) {
                capacity *= 2;
                AdaptiveTask* grown = realloc(stack, capacity * sizeof(AdaptiveTask));
                if (grown == NULL) {
                    free(stack);
                    fprintf(stderr, "Error: failed to grow adaptive stack\n");
                    exit(EXIT_FAILURE);
                }
                stack = grown;
            }
            stack[top++] = (AdaptiveTask){m, task.r, right_val, task.depth + 1};
            stack[top++] = (AdaptiveTask){task.l, m, left_val, task.depth + 1};
        }
    }

    free(stack);
    return total;
}
