#ifndef CLI_H
#define CLI_H

#include "common/integration_config.h"

int parse_cli(int argc, char** argv, IntegrationConfig* cfg);
const char* integration_mode_name(IntegrationMode mode);
void print_csv_header(void);
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
);

#endif
