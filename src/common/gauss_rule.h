#ifndef GAUSS_RULE_H
#define GAUSS_RULE_H

typedef struct{
    int order;
    double* nodes;
    double* weights;
} GaussRule ;

void gauss_rule_init(GaussRule *rule, int order);
void gauss_rule_free(GaussRule *rule);

#endif
