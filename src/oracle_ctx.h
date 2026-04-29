#ifndef ORACLE_CTX_H
#define ORACLE_CTX_H

#include "opt.h"

typedef struct {
    opt_obj_fn      obj;
    opt_grad_fn     grad;          /* NULL for gradient-free methods */
    void           *userdata;
    unsigned short  n;             /* parameter dimension */
    unsigned int    n_calls;       /* total oracle invocations recorded = history length */
    opt_record_t   *history;       /* record array */
    unsigned int    cap;           /* allocated capacity */
    void           *nlopt_handle;  /* nlopt_opt, typed void* to avoid NLopt in header */
} oracle_ctx_t;

#ifdef __cplusplus
extern "C" {
#endif

int  oracle_ctx_init(oracle_ctx_t *ctx, opt_obj_fn obj, opt_grad_fn grad,
                     void *userdata, unsigned short n);
void oracle_ctx_free(oracle_ctx_t *ctx);
void oracle_ctx_release(oracle_ctx_t *ctx);

/* NLopt-facing callback; registered via nlopt_set_min_objective */
double opt_nlopt_adapter(unsigned n, const double *x, double *grad_out, void *data);

#ifdef __cplusplus
}
#endif

#endif /* ORACLE_CTX_H */
