#ifndef OPT_H
#define OPT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =====================================================================================================================
 *                                                  type definitions
 * =====================================================================================================================
 */

/*
 * Status codes returned by all public functions.
 */
typedef enum {
    OPT_OK = 0,
    OPT_WARN_MAXCALLS,
    OPT_ERR_NULL_PTR,
    OPT_ERR_INVALID_CONFIG,
    OPT_ERR_NLOPT_FAILURE,
    OPT_ERR_ALLOC,
} opt_status_t;

/*
 * Optimisation method selector.  All methods are NLopt-backed.
 */
typedef enum {
    OPT_NLOPT_LBFGS = 0,
    OPT_NLOPT_BOBYQA,
    OPT_NLOPT_COBYLA,
    OPT_NLOPT_NELDERMEAD,
    OPT_NLOPT_DIRECT,
    OPT_NLOPT_MMA,
    OPT_NLOPT_SLSQP,
    OPT_NLOPT_TNEWTON,
    OPT_METHOD_COUNT
} opt_method_t;

/*
 * Oracle callbacks.  Each invocation counts as one oracle call.
 * n is the parameter dimension (<= USHRT_MAX).
 */
typedef double (*opt_obj_fn)(unsigned short n, const double *params, void *userdata);
typedef void   (*opt_grad_fn)(unsigned short n, const double *params,
                              double *grad_out, void *userdata);

/*
 * Single oracle-call record.
 * grad_norm is NaN when the gradient was not evaluated on this call.
 */
typedef struct {
    unsigned int  step_idx;
    unsigned int  call_idx;
    double        obj_value;
    double        grad_norm;
} opt_record_t;

/*
 * Solver configuration.
 * Zero-initialised fields are replaced with per-method defaults inside opt_run.
 */
typedef struct {
    double       abs_obj;
    double       rel_obj;
    double       abs_params;
    double       rel_params;
    unsigned int max_calls;
    unsigned int max_steps;
} opt_config_t;

/*
 * Library-owned result.
 * Caller frees via opt_free(); never free individual fields directly.
 */
typedef struct {
    opt_status_t  status;
    double       *params;
    double        value;
    unsigned int  n_steps;
    unsigned int  n_calls;
    opt_record_t *history;
} opt_result_t;

/*
 * =====================================================================================================================
 *                                                  public API
 * =====================================================================================================================
 */

/*
 * @brief   Run an optimisation using the specified method.
 *
 * @param[in]  method      Algorithm selector.
 * @param[in]  n           Parameter dimension (> 0).
 * @param[in,out] params   Initial parameter vector on entry; optimal parameters on exit
 *                         (written only when status == OPT_OK or OPT_WARN_MAXCALLS).
 * @param[in]  obj         Objective function callback; must not be NULL.
 * @param[in]  grad        Gradient callback; may be NULL for gradient-free methods.
 * @param[in]  userdata    Passed verbatim to obj and grad.
 * @param[in]  config      Solver configuration; NULL uses per-method defaults.
 * @param[out] result_out  Receives a pointer to a library-owned opt_result_t.
 *                         Must point to NULL on entry.  Caller frees via opt_free().
 *
 * @return  OPT_OK on success; OPT_WARN_MAXCALLS if max_calls was reached before
 *          convergence; other OPT_ERR_* codes on failure.
 */
opt_status_t opt_run(opt_method_t method, unsigned short n, double *params,
                     opt_obj_fn obj, opt_grad_fn grad, void *userdata,
                     const opt_config_t *config, opt_result_t **result_out);

/*
 * @brief   Free a result previously returned by opt_run.  Safe to call with NULL.
 */
void opt_free(opt_result_t *result);

/*
 * @brief   Return a human-readable string for a status code.
 *          The returned pointer is to a string literal; do not free it.
 */
const char *opt_status_str(opt_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* OPT_H */
