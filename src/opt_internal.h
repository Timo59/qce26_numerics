#ifndef OPT_INTERNAL_H
#define OPT_INTERNAL_H

#include "opt.h"

/* Methods that require a non-NULL gradient callback.
 * Must stay consistent with method_map in opt_nlopt.c. */
static inline int opt_method_needs_grad(opt_method_t m)
{
    static const int table[OPT_METHOD_COUNT] = {
        [OPT_NLOPT_LBFGS]   = 1,
        [OPT_NLOPT_MMA]     = 1,
        [OPT_NLOPT_SLSQP]   = 1,
        [OPT_NLOPT_TNEWTON] = 1,
    };
    return (m < OPT_METHOD_COUNT) ? table[m] : 0;
}

#endif /* OPT_INTERNAL_H */
