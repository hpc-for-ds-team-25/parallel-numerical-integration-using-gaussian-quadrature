#ifndef INTEGRATION_CONFIG_H
#define INTEGRATION_CONFIG_H

typedef struct {
    double a, b;  // integration interval
    long long subintervals_count;
    int order;
    int function_id; // to switch integration function on the fly
} IntegrationConfig;

#endif
