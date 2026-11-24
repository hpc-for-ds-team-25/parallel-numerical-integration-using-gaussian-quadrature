#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "common/gauss_rule.h"

int main (int argc, char** argv) {
    GaussRule gauss_rule;
    gauss_rule_init(&gauss_rule, 3);

    // todo: actually use the nodes and weights

    gauss_rule_free(&gauss_rule);
    return 0;
}
