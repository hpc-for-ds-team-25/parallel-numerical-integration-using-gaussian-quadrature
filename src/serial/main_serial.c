#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "common/cli.h"
#include "common/gauss_rule.h"
#include "common/integrands.h"
#include "common/integration_config.h"
#include "common/timer.h"

double integrate_1d_composite(
    Integrand1DFn func,
    double a,
    double b,
    long long num_panels,
    const GaussRule* rule
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

int main(int argc, char** argv) {
    IntegrationConfig cfg;
    int cli_status = parse_cli(argc, argv, &cfg);
    if (cli_status != 0) {
        return cli_status > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const Integrand1D* integ_1d = NULL;
    const Integrand3D* integ_3d = NULL;
    const char* func_name = NULL;
    double exact_value = 0.0;

    if (cfg.mode == MODE_3D) {
        integ_3d = get_integrand_3d(cfg.function_id);
        func_name = integ_3d->name;
        exact_value = integ_3d->exact_value;
    } else {
        integ_1d = get_integrand_1d(cfg.function_id);
        func_name = integ_1d->name;
        exact_value = integ_1d->exact_value;
    }

    if (cfg.print_header) {
        print_csv_header();
    }

    for (int run = 1; run <= cfg.repeats; ++run) {
        GaussRule rule;
        double t0 = wall_time();
        gauss_rule_init(&rule, cfg.order);
        double t1 = wall_time();

        double result = 0.0;
        if (cfg.mode == MODE_1D) {
            result = integrate_1d_composite(
                integ_1d->integrand_fn, cfg.a, cfg.b, cfg.subintervals_count, &rule
            );
        } else if (cfg.mode == MODE_3D) {
            result = integrate_3d_composite(
                integ_3d->integrand_fn, cfg.a, cfg.b, cfg.subintervals_count, &rule
            );
        } else {
            result = integrate_1d_adaptive(
                integ_1d->integrand_fn, cfg.a, cfg.b, cfg.subintervals_count,
                cfg.tol, cfg.max_depth, &rule
            );
        }
        double t2 = wall_time();

        double abs_error = fabs(result - exact_value);
        print_csv_row("serial", &cfg, func_name, 1, 1, run, t1 - t0, t2 - t1, result, abs_error);

        gauss_rule_free(&rule);
    }

    return EXIT_SUCCESS;
}

double integrate_1d_composite(
    Integrand1DFn func,
    double a,
    double b,
    long long num_panels,
    const GaussRule* rule
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
