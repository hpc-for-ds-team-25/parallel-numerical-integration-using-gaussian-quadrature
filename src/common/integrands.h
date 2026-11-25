#ifndef INTEGRANDS_H
#define INTEGRANDS_H

typedef double (*Integrand1DFn)(double x);
typedef double (*Integrannd3DFn)(double x, double y, double z);

typedef enum {
    FUNC_POLY,
    FUNC_SMOOTH_1D,
    FUNC_OSC_1D,
    FUNC_PEAK_1D,
} FunctionId;

typedef struct {
    FunctionId id;
    const char* name;
    Integrand1DFn integrand_fn;
    double default_a;
    double default_b;
    double exact_value;
} Integrand1D;

const Integrand1D *integrant_1d_get(FunctionId id);

#endif
