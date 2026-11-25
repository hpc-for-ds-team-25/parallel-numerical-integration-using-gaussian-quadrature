#include "integrands.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double f_poly(double x) {
    double x2 = x * x;
    double x4 = x2 * x2;
    double x8 = x4 * x4;
    double x15 = x8 * x4 * x2 * x;
    return x15 - 3.0 * x8 + 2.0 * x2 + 1.0;
}

static double f_smooth_1d(double x) {
    return exp(-x) * sin(5.0 * x);
}

static double f_osc_1d(double x) {
    return exp(-0.1 * x) * sin(50.0 * x) + exp(-0.2 * x) * cos(120.0 * x);
}

static double f_peak_1d(double x) {
    const double sigma = 1e-4;
    const double x0 = 0.2;
    const double norm = 1.0 / (sigma * sqrt(2.0 * M_PI));
    double z = (x - x0) / sigma;
    return norm * exp(-0.5 * z * z);
}

static double f_heavy_3d(double x, double y, double z) {
    return (exp(-x) * sin(M_PI * x))
         * (exp(-y) * sin(M_PI * y))
         * (exp(-z) * sin(M_PI * z));
}

static const Integrand1D table_1d[] = {
    {FUNC_POLY,      "poly",      f_poly,      -1.0,  1.0, 8.0 / 3.0},
    {FUNC_SMOOTH_1D, "smooth_1d", f_smooth_1d,  0.0, 10.0, 0.19229972557116717},
    {FUNC_OSC_1D,    "osc_1d",    f_osc_1d,     0.0, 20.0, 0.018460050968047276},
    {FUNC_PEAK_1D,   "peak_1d",   f_peak_1d,    0.0,  1.0, 1.0}
};

static const Integrand3D table_3d[] = {
    {FUNC_HEAVY_3D, "heavy_3d", f_heavy_3d, 0.0, 1.0, 0.061794791353418359}
};

const Integrand1D* get_integrand_1d(FunctionId id) {
    size_t count = sizeof(table_1d) / sizeof(table_1d[0]);
    for (size_t i = 0; i < count; ++i) {
        if (table_1d[i].id == id) {
            return &table_1d[i];
        }
    }
    return NULL;
}

const Integrand3D* get_integrand_3d(FunctionId id) {
    size_t count = sizeof(table_3d) / sizeof(table_3d[0]);
    for (size_t i = 0; i < count; ++i) {
        if (table_3d[i].id == id) {
            return &table_3d[i];
        }
    }
    return NULL;
}

int get_function_id_by_name(const char* name, FunctionId* out_id) {
    if (name == NULL || out_id == NULL) {
        return -1;
    }

    size_t count_1d = sizeof(table_1d) / sizeof(table_1d[0]);
    for (size_t i = 0; i < count_1d; ++i) {
        if (strcmp(table_1d[i].name, name) == 0) {
            *out_id = table_1d[i].id;
            return 0;
        }
    }

    size_t count_3d = sizeof(table_3d) / sizeof(table_3d[0]);
    for (size_t i = 0; i < count_3d; ++i) {
        if (strcmp(table_3d[i].name, name) == 0) {
            *out_id = table_3d[i].id;
            return 0;
        }
    }

    return -1;
}
