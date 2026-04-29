#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <nlopt.h>
#include "oracle_ctx.h"

/* Initial allocation capacity for history */
#define ORACLE_CTX_INIT_CAP 64

/* -------------------------------------------------------------------------
 * oracle_ctx_grow — double capacity of history array.
 *
 * Returns 0 on success, -1 on allocation failure.
 * On failure, the caller (record_call) will call nlopt_force_stop.
 * -------------------------------------------------------------------------
 */
static int oracle_ctx_grow(oracle_ctx_t *ctx)
{
    unsigned int new_cap = ctx->cap * 2;

    opt_record_t *new_history = realloc(ctx->history, new_cap * sizeof(opt_record_t));
    if (new_history == NULL) {
        return -1;
    }
    ctx->history = new_history;
    ctx->cap = new_cap;
    return 0;
}

/* -------------------------------------------------------------------------
 * record_call — append one oracle-call record to the history.
 *
 * Grows the history if needed. On grow failure, calls nlopt_force_stop to
 * abort the current optimisation run cleanly; the record is silently dropped.
 * -------------------------------------------------------------------------
 */
static void record_call(oracle_ctx_t *ctx, double obj_val, double grad_norm)
{
    if (ctx->n_calls >= ctx->cap) {
        if (oracle_ctx_grow(ctx) != 0) {
            /* Growth failed — signal NLopt to stop; record is dropped */
            nlopt_force_stop((nlopt_opt)ctx->nlopt_handle);
            return;
        }
    }

    unsigned int idx = ctx->n_calls;

    ctx->history[idx].step_idx  = 0;               /* NLopt does not expose accept/reject */
    ctx->history[idx].call_idx  = idx;
    ctx->history[idx].obj_value = obj_val;
    ctx->history[idx].grad_norm = grad_norm;

    ctx->n_calls++;
}

/* -------------------------------------------------------------------------
 * oracle_ctx_init — initialise an oracle_ctx_t with initial capacity 64.
 *
 * Returns 0 on success, -1 on allocation failure.
 * -------------------------------------------------------------------------
 */
int oracle_ctx_init(oracle_ctx_t *ctx, opt_obj_fn obj, opt_grad_fn grad,
                    void *userdata, unsigned short n)
{
    /* Zero-initialise all fields first */
    memset(ctx, 0, sizeof(oracle_ctx_t));

    ctx->obj      = obj;
    ctx->grad     = grad;
    ctx->userdata = userdata;
    ctx->n        = n;

    if (n == 0) return -1;

    ctx->history = malloc(ORACLE_CTX_INIT_CAP * sizeof(opt_record_t));
    if (ctx->history == NULL) {
        return -1;
    }

    ctx->cap = ORACLE_CTX_INIT_CAP;
    return 0;
}

/* -------------------------------------------------------------------------
 * oracle_ctx_free — release all memory owned by the context.
 * -------------------------------------------------------------------------
 */
void oracle_ctx_free(oracle_ctx_t *ctx)
{
    free(ctx->history);
    ctx->history = NULL;
}

/* -------------------------------------------------------------------------
 * oracle_ctx_release — null out history WITHOUT freeing it.
 *
 * Called after ownership of the history allocation has been transferred to
 * opt_result_t, to prevent a subsequent oracle_ctx_free from double-freeing.
 * -------------------------------------------------------------------------
 */
void oracle_ctx_release(oracle_ctx_t *ctx)
{
    ctx->history = NULL;
}

/* -------------------------------------------------------------------------
 * opt_nlopt_adapter — NLopt-facing callback registered via
 * nlopt_set_min_objective.
 *
 * Each invocation produces exactly one oracle-call record containing both the
 * objective value and the gradient norm (or NaN if gradient was not evaluated).
 * -------------------------------------------------------------------------
 */
double opt_nlopt_adapter(unsigned n, const double *x, double *grad_out, void *data)
{
    (void)n;
    oracle_ctx_t *ctx = data;

    double val = ctx->obj(ctx->n, x, ctx->userdata);
    double gnorm = (double)NAN;

    if (grad_out != NULL && ctx->grad != NULL) {
        ctx->grad(ctx->n, x, grad_out, ctx->userdata);
        gnorm = 0.0;
        for (unsigned int i = 0; i < ctx->n; i++) {
            gnorm += grad_out[i] * grad_out[i];
        }
        gnorm = sqrt(gnorm);
    }

    record_call(ctx, val, gnorm);
    return val;
}
