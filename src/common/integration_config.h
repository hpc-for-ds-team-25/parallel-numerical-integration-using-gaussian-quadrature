#ifndef INTEGRATION_CONFIG_H
#define INTEGRATION_CONFIG_H

#include "common/integrands.h"

typedef enum {
    MODE_1D = 0,
    MODE_3D,
    MODE_ADAPTIVE
} IntegrationMode;

typedef struct {
    IntegrationMode mode;
    FunctionId function_id;
    double a;
    double b;
    long long subintervals_count;
    int order;
    int repeats;
    double tol;
    int max_depth;
    int print_header;
} IntegrationConfig;

#endif
