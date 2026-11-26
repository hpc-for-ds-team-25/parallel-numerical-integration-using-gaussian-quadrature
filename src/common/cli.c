#include "common/cli.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "Options:\n"
        "  --mode <1d|3d|adaptive>   Integration mode (default: 1d)\n"
        "  --func <name>             Integrand: poly, smooth_1d, osc_1d, peak_1d, heavy_3d\n"
        "  --order <n>               Gauss-Legendre points per panel (default: 8)\n"
        "  --panels <M>              Panels (1D/adaptive initial) or panels/axis (3D)\n"
        "  --repeats <R>             Number of timed runs (default: 1)\n"
        "  --a <val>                 Override interval start a\n"
        "  --b <val>                 Override interval end b\n"
        "  --tol <eps>               Adaptive tolerance (default: 1e-12)\n"
        "  --max-depth <d>           Adaptive max split depth (default: 30)\n"
        "  --no-header               Omit CSV header line\n"
        "  -h, --help                Show this help message\n",
        prog
    );
}

static int parse_int(const char* s, int* out) {
    char* end = NULL;
    errno = 0;
    long val = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || val < INT_MIN || val > INT_MAX) {
        return -1;
    }
    *out = (int)val;
    return 0;
}

static int parse_ll(const char* s, long long* out) {
    char* end = NULL;
    errno = 0;
    long long val = strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return -1;
    }
    *out = val;
    return 0;
}

static int parse_double(const char* s, double* out) {
    char* end = NULL;
    errno = 0;
    double val = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0') {
        return -1;
    }
    *out = val;
    return 0;
}

const char* integration_mode_name(IntegrationMode mode) {
    switch (mode) {
        case MODE_1D:       return "1d";
        case MODE_3D:       return "3d";
        case MODE_ADAPTIVE: return "adaptive";
        default:            return "unknown";
    }
}

int parse_cli(int argc, char** argv, IntegrationConfig* cfg) {
    cfg->mode = MODE_1D;
    cfg->function_id = FUNC_POLY;
    cfg->a = 0.0;
    cfg->b = 0.0;
    cfg->subintervals_count = 0;
    cfg->order = 8;
    cfg->repeats = 1;
    cfg->tol = 1e-12;
    cfg->max_depth = 30;
    cfg->print_header = 1;

    int has_func = 0;
    int has_panels = 0;
    int has_a = 0;
    int has_b = 0;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 1;
        } else if (strcmp(arg, "--no-header") == 0) {
            cfg->print_header = 0;
        } else if (strcmp(arg, "--mode") == 0 && i + 1 < argc) {
            const char* val = argv[++i];
            if (strcmp(val, "1d") == 0) {
                cfg->mode = MODE_1D;
            } else if (strcmp(val, "3d") == 0) {
                cfg->mode = MODE_3D;
            } else if (strcmp(val, "adaptive") == 0) {
                cfg->mode = MODE_ADAPTIVE;
            } else {
                fprintf(stderr, "Error: unknown mode '%s'\n", val);
                return -1;
            }
        } else if (strcmp(arg, "--func") == 0 && i + 1 < argc) {
            const char* val = argv[++i];
            if (get_function_id_by_name(val, &cfg->function_id) != 0) {
                fprintf(stderr, "Error: unknown function '%s'\n", val);
                return -1;
            }
            has_func = 1;
        } else if (strcmp(arg, "--order") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &cfg->order) != 0) {
                fprintf(stderr, "Error: invalid --order value\n");
                return -1;
            }
        } else if (strcmp(arg, "--panels") == 0 && i + 1 < argc) {
            if (parse_ll(argv[++i], &cfg->subintervals_count) != 0) {
                fprintf(stderr, "Error: invalid --panels value\n");
                return -1;
            }
            has_panels = 1;
        } else if (strcmp(arg, "--repeats") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &cfg->repeats) != 0) {
                fprintf(stderr, "Error: invalid --repeats value\n");
                return -1;
            }
        } else if (strcmp(arg, "--a") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &cfg->a) != 0) {
                fprintf(stderr, "Error: invalid --a value\n");
                return -1;
            }
            has_a = 1;
        } else if (strcmp(arg, "--b") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &cfg->b) != 0) {
                fprintf(stderr, "Error: invalid --b value\n");
                return -1;
            }
            has_b = 1;
        } else if (strcmp(arg, "--tol") == 0 && i + 1 < argc) {
            if (parse_double(argv[++i], &cfg->tol) != 0) {
                fprintf(stderr, "Error: invalid --tol value\n");
                return -1;
            }
        } else if (strcmp(arg, "--max-depth") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &cfg->max_depth) != 0) {
                fprintf(stderr, "Error: invalid --max-depth value\n");
                return -1;
            }
        } else {
            fprintf(stderr, "Error: unknown or incomplete option '%s'\n", arg);
            print_usage(argv[0]);
            return -1;
        }
    }

    if (!has_func) {
        if (cfg->mode == MODE_3D) {
            cfg->function_id = FUNC_HEAVY_3D;
        } else if (cfg->mode == MODE_ADAPTIVE) {
            cfg->function_id = FUNC_PEAK_1D;
        } else {
            cfg->function_id = FUNC_POLY;
        }
    }

    if (cfg->mode == MODE_3D) {
        const Integrand3D* integ = get_integrand_3d(cfg->function_id);
        if (integ == NULL) {
            fprintf(stderr, "Error: selected function is not a 3D integrand (use --func heavy_3d)\n");
            return -1;
        }
        if (!has_a) cfg->a = integ->default_a;
        if (!has_b) cfg->b = integ->default_b;
        if (!has_panels) cfg->subintervals_count = 16;
    } else {
        const Integrand1D* integ = get_integrand_1d(cfg->function_id);
        if (integ == NULL) {
            fprintf(stderr, "Error: selected function is not a 1D integrand\n");
            return -1;
        }
        if (!has_a) cfg->a = integ->default_a;
        if (!has_b) cfg->b = integ->default_b;
        if (!has_panels) {
            cfg->subintervals_count = (cfg->function_id == FUNC_POLY && cfg->mode == MODE_1D) ? 1 : 1000;
        }
    }

    if (cfg->order < 1 || cfg->subintervals_count < 1 || cfg->repeats < 1 ||
        cfg->a >= cfg->b || cfg->tol <= 0.0 || cfg->max_depth < 1) {
        fprintf(stderr, "Error: invalid parameter range\n");
        return -1;
    }

    return 0;
}

void print_csv_header(void) {
    printf("impl,mode,func,order,panels,ranks,threads,run_id,t_nodes_s,t_eval_s,t_total_s,result,abs_error\n");
}

void print_csv_row(
    const char* impl,
    const IntegrationConfig* cfg,
    const char* func_name,
    int ranks,
    int threads,
    int run_id,
    double t_nodes,
    double t_eval,
    double result,
    double abs_error
) {
    printf("%s,%s,%s,%d,%lld,%d,%d,%d,%.6f,%.6f,%.6f,%.15f,%.3e\n",
        impl,
        integration_mode_name(cfg->mode),
        func_name,
        cfg->order,
        cfg->subintervals_count,
        ranks,
        threads,
        run_id,
        t_nodes,
        t_eval,
        t_nodes + t_eval,
        result,
        abs_error
    );
}
