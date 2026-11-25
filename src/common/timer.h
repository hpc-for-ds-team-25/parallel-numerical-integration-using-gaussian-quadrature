#ifndef TIMER_H
#define TIMER_H

#include <stddef.h>
#include <sys/time.h>

static inline double wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

#endif
