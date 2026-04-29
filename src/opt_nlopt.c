/* opt_nlopt.c — NLopt backend for opt_run.
 *
 * This translation unit is internal. It owns:
 *   - Two co-indexed static tables (method_map, method_needs_bounds).
 *   - opt_nlopt_run(), which creates an NLopt optimizer, wires in the oracle
 *     adapter, drives the optimisation, and maps NLopt result codes to
 *     opt_status_t.
 *
 * The method_needs_grad predicate lives in opt_internal.h so that opt_run.c
 * can share it without duplication.
 *
 * No exit(), no printf, no assert(). All errors propagate via return values.
 */

#include <limits.h>
#include <math.h>
#include <nlopt.h>
#include "opt.h"
#include "oracle_ctx.h"
#include "opt_internal.h"

/* -------------------------------------------------------------------------
 * Method tables — co-indexed by opt_method_t.
 * -------------------------------------------------------------------------
 */

/* Maps each opt_method_t to its NLopt algorithm enum value. */
static const nlopt_algorithm method_map[OPT_METHOD_COUNT] = {
    [OPT_NLOPT_LBFGS]      = NLOPT_LD_LBFGS,
    [OPT_NLOPT_BOBYQA]     = NLOPT_LN_BOBYQA,
    [OPT_NLOPT_COBYLA]     = NLOPT_LN_COBYLA,
    [OPT_NLOPT_NELDERMEAD] = NLOPT_LN_NELDERMEAD,
    [OPT_NLOPT_DIRECT]     = NLOPT_GN_DIRECT_L,
    [OPT_NLOPT_MMA]        = NLOPT_LD_MMA,
    [OPT_NLOPT_SLSQP]      = NLOPT_LD_SLSQP,
    [OPT_NLOPT_TNEWTON]    = NLOPT_LD_TNEWTON,
};

/*
 * Methods that require finite box bounds.  DIRECT and BOBYQA will
 * malfunction or fail without them; we supply [-π, π] as a conservative
 * default suitable for variational angle parameters.
 */
static const int method_needs_bounds[OPT_METHOD_COUNT] = {
    [OPT_NLOPT_DIRECT] = 1,
    [OPT_NLOPT_BOBYQA] = 1,
};

/* -------------------------------------------------------------------------
 * opt_nlopt_run — internal entry point; not exported.
 *
 * @param method        Algorithm selector; must be < OPT_METHOD_COUNT.
 * @param n             Parameter dimension; must be > 0.
 * @param params        In/out: starting point on entry, optimum on exit
 *                      (mutated in-place by nlopt_optimize).
 * @param cfg           Fully-merged configuration (no zero fields).
 * @param ctx           Pre-initialised oracle context; ctx->nlopt_handle is
 *                      set here so that record_call can call nlopt_force_stop.
 * @param opt_value_out Receives NLopt's reported minimum objective value.
 *
 * @return  OPT_OK                  on convergence
 *          OPT_WARN_MAXCALLS       if max oracle calls / time was reached
 *          OPT_ERR_ALLOC           if nlopt_create failed, out-of-memory, or
 *                                  nlopt_force_stop was triggered by a grow
 *                                  failure inside record_call
 *          OPT_ERR_NLOPT_FAILURE   for any other NLopt error code
 * -------------------------------------------------------------------------
 */
opt_status_t opt_nlopt_run(opt_method_t method, unsigned short n,
                           double *params, const opt_config_t *cfg,
                           oracle_ctx_t *ctx, double *opt_value_out)
{
    nlopt_opt opt = nlopt_create(method_map[method], (unsigned)n);
    if (opt == NULL) {
        return OPT_ERR_ALLOC;
    }

    ctx->nlopt_handle = opt;

    if (method_needs_bounds[method]) {
        nlopt_set_lower_bounds1(opt, -M_PI);
        nlopt_set_upper_bounds1(opt,  M_PI);
    }

    nlopt_set_ftol_abs(opt, cfg->abs_obj);
    nlopt_set_ftol_rel(opt, cfg->rel_obj);
    nlopt_set_xtol_abs1(opt, cfg->abs_params);
    nlopt_set_xtol_rel(opt, cfg->rel_params);
    nlopt_set_maxeval(opt,
        cfg->max_calls > (unsigned)INT_MAX ? INT_MAX : (int)cfg->max_calls);

    nlopt_set_min_objective(opt, opt_nlopt_adapter, ctx);

    *opt_value_out = 0.0;
    nlopt_result rc = nlopt_optimize(opt, params, opt_value_out);

    nlopt_destroy(opt);
    ctx->nlopt_handle = NULL;

    switch (rc) {
        case NLOPT_SUCCESS:
        case NLOPT_FTOL_REACHED:
        case NLOPT_XTOL_REACHED:
        case NLOPT_STOPVAL_REACHED:
            return OPT_OK;

        case NLOPT_MAXEVAL_REACHED:
        case NLOPT_MAXTIME_REACHED:
        case NLOPT_ROUNDOFF_LIMITED:
            return OPT_WARN_MAXCALLS;

        case NLOPT_OUT_OF_MEMORY:
        case NLOPT_FORCED_STOP:
            return OPT_ERR_ALLOC;

        default:
            return OPT_ERR_NLOPT_FAILURE;
    }
}
