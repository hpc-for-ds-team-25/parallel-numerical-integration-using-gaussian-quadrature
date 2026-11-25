#include <stdlib.h>

#include "integrands.h"


static double f_poly(double x) {
    double x2 = x * x;
    double x4 = x2 * x2;
    double x8 = x4 * x4;
    double x15 = x8 * x4 * x2 * x;
    return x15 - 3.0 * x8 + 2.0 * x2 + 1.0;
}

static const Integrand1D table_1d[] = {
    {FUNC_POLY, "poly", f_poly, -1.0, 1.0, 8.0/3.0}
};

const Integrand1D* get_integrand_1d(FunctionId id) {
    if (id <= 0 || id > (int) (sizeof(table_1d) / sizeof(table_1d[0]))) {
        return NULL;
    }

    return &table_1d[id];
}
