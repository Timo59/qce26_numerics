/* opt_run.c — public entry point for the unified optimiser interface.
 *
 * Exports: opt_run(), opt_free(), opt_status_str().
 * All other symbols in this file are file-scoped (static).
 *
 * No exit(), no printf, no assert(). All errors propagate via opt_status_t.
 */

#include <stdlib.h>
#include <string.h>
#include "opt.h"
#include "oracle_ctx.h"
#include "opt_internal.h"

/* Forward declaration of the internal NLopt backend (defined in opt_nlopt.c). */
opt_status_t opt_nlopt_run(opt_method_t method, unsigned short n,
                           double *params, const opt_config_t *cfg,
                           oracle_ctx_t *ctx, double *opt_value_out);

/* -------------------------------------------------------------------------
 * Per-method default configuration table.
 *
 * All tolerance fields are non-zero so that merge_config's zero-override
 * logic works correctly.  max_steps is 0 for all NLopt methods because the
 * step count is not observable through NLopt's API.
 * -------------------------------------------------------------------------
 */
static const opt_config_t method_defaults[OPT_METHOD_COUNT] = {
    [OPT_NLOPT_LBFGS]      = {1e-9, 1e-6, 1e-9, 1e-6, 10000, 0},
    [OPT_NLOPT_BOBYQA]     = {1e-9, 1e-6, 1e-9, 1e-6,  5000, 0},
    [OPT_NLOPT_COBYLA]     = {1e-9, 1e-6, 1e-9, 1e-6,  5000, 0},
    [OPT_NLOPT_NELDERMEAD] = {1e-9, 1e-6, 1e-9, 1e-6,  5000, 0},
    [OPT_NLOPT_DIRECT]     = {1e-9, 1e-6, 1e-9, 1e-6, 50000, 0},
    [OPT_NLOPT_MMA]        = {1e-9, 1e-6, 1e-9, 1e-6, 10000, 0},
    [OPT_NLOPT_SLSQP]      = {1e-9, 1e-6, 1e-9, 1e-6, 10000, 0},
    [OPT_NLOPT_TNEWTON]    = {1e-9, 1e-6, 1e-9, 1e-6, 10000, 0},
};

/* -------------------------------------------------------------------------
 * merge_config — produce a fully-specified config for the given method.
 *
 * Starts from the per-method defaults and overrides each field with the
 * corresponding user-supplied value, but only when that value is non-zero.
 * A NULL user config leaves all fields at their defaults.
 * -------------------------------------------------------------------------
 */
static opt_config_t merge_config(opt_method_t method, const opt_config_t *user)
{
    opt_config_t cfg = method_defaults[method];
    if (user == NULL) {
        return cfg;
    }
    if (user->abs_obj    != 0.0) cfg.abs_obj    = user->abs_obj;
    if (user->rel_obj    != 0.0) cfg.rel_obj    = user->rel_obj;
    if (user->abs_params != 0.0) cfg.abs_params = user->abs_params;
    if (user->rel_params != 0.0) cfg.rel_params = user->rel_params;
    if (user->max_calls  != 0)   cfg.max_calls  = user->max_calls;
    if (user->max_steps  != 0)   cfg.max_steps  = user->max_steps;
    return cfg;
}

/* -------------------------------------------------------------------------
 * opt_run — public entry point.
 * -------------------------------------------------------------------------
 */
opt_status_t opt_run(opt_method_t method, unsigned short n, double *params,
                     opt_obj_fn obj, opt_grad_fn grad, void *userdata,
                     const opt_config_t *config, opt_result_t **result_out)
{
    if (params == NULL || obj == NULL || result_out == NULL) {
        return OPT_ERR_NULL_PTR;
    }

    if (n == 0 || method >= OPT_METHOD_COUNT) {
        return OPT_ERR_INVALID_CONFIG;
    }

    if (opt_method_needs_grad(method) && grad == NULL) {
        return OPT_ERR_INVALID_CONFIG;
    }

    if (*result_out != NULL) {
        return OPT_ERR_INVALID_CONFIG;
    }

    opt_config_t cfg = merge_config(method, config);

    *result_out = calloc(1, sizeof(opt_result_t));
    if (*result_out == NULL) {
        return OPT_ERR_ALLOC;
    }

    oracle_ctx_t ctx;
    if (oracle_ctx_init(&ctx, obj, grad, userdata, n) != 0) {
        free(*result_out);
        *result_out = NULL;
        return OPT_ERR_ALLOC;
    }

    opt_status_t s = opt_nlopt_run(method, n, params, &cfg, &ctx, &(*result_out)->value);

    (*result_out)->params = malloc((size_t)n * sizeof(double));
    if ((*result_out)->params == NULL) {
        oracle_ctx_free(&ctx);
        free(*result_out);
        *result_out = NULL;
        return OPT_ERR_ALLOC;
    }
    memcpy((*result_out)->params, params, (size_t)n * sizeof(double));

    (*result_out)->status  = s;
    (*result_out)->n_steps = 0;
    (*result_out)->n_calls = ctx.n_calls;
    (*result_out)->history = ctx.history;

    oracle_ctx_release(&ctx);

    return s;
}

void opt_free(opt_result_t *r)
{
    if (r == NULL) {
        return;
    }
    free(r->params);
    free(r->history);
    free(r);
}

const char *opt_status_str(opt_status_t status)
{
    switch (status) {
        case OPT_OK:                 return "OPT_OK";
        case OPT_WARN_MAXCALLS:      return "OPT_WARN_MAXCALLS";
        case OPT_ERR_NULL_PTR:       return "OPT_ERR_NULL_PTR";
        case OPT_ERR_INVALID_CONFIG: return "OPT_ERR_INVALID_CONFIG";
        case OPT_ERR_NLOPT_FAILURE:  return "OPT_ERR_NLOPT_FAILURE";
        case OPT_ERR_ALLOC:          return "OPT_ERR_ALLOC";
        default:                     return "unknown status";
    }
}
