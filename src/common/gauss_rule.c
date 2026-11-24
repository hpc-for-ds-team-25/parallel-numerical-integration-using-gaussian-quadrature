#include "gauss_rule.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void gauss_rule_init(GaussRule *rule, int order) {
    rule->order = order;
    rule->nodes = malloc(order * sizeof(double));
    rule->weights = malloc(order * sizeof(double));

    int m = (order + 1) / 2;
    for (int i = 1; i <= m; ++i) {
        double root = cos(M_PI * ((i - 0.25)/(order + 0.5)));
        double dp = 0.0;
        double droot = 0.0;

        do {
            double p_prev = 0.0;
            double p_curr = 1.0;
            for (int k = 1; k <= order; ++k) {
                double p = (((2*k-1) * root * p_curr) - (k-1) * p_prev)/k;
                p_prev = p_curr;
                p_curr = p;
            }

            dp = order * (root * p_curr - p_prev) / (root * root - 1.0);
            droot = p_curr / dp;
            root -= droot;
        } while (fabs(droot) >= 1e-15);

        double weight = 2.0 / ((1.0 - (root*root))*(dp*dp));

        rule->nodes[i - 1] = -root;
        rule->weights[i - 1] = weight;
        rule->nodes[order - i] = root;
        rule->weights[order - i] = weight;
    }
}

void gauss_rule_free(GaussRule *rule) {
    free(rule->nodes);
    free(rule->weights);

    rule->order = 0;
    rule->nodes = NULL;
    rule->weights = NULL;
}
