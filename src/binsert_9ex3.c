#include "opt.h"
#include <orkan.h>

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * =====================================================================================================================
 * MACROS
 * =====================================================================================================================
 */

#define N_CITIES  9
#define NUM_REG   (N_CITIES - 1)
#define LEN_REG   3                          /* ceil(log2(NUM_REG)) -- update manually when changing N_CITIES */
#define N_QUBITS  (NUM_REG * LEN_REG)
#define DEPTH     1
#define OPTIMUM   25.0

/* Upper bound on the number of binsert operators: sum of ceil(log2(m)) for
 * m = 2..NUM_REG.  The exact count is determined at runtime by build_bin_ops(). */
#define N_BIN_OPS_UPPER  (NUM_REG * LEN_REG)

/* Upper bound on total qubit-SWAP pairs across all operators.  Each operator
 * may contain up to ceil(NUM_REG/2) disjoint register transpositions, each
 * expanding to LEN_REG qubit pairs. */
#define N_BIN_PAIRS_UPPER  (N_BIN_OPS_UPPER * NUM_REG * LEN_REG)

/*
 * =====================================================================================================================
 * INSTANCE
 * =====================================================================================================================
 */

/* Asymmetric distance matrix for 9 cities.
 * Stored as a flat N_CITIES x N_CITIES row-major array.  dist[i * N_CITIES + j]
 * is the travel cost from city i to city j.  Works for asymmetric instances. */
static const double dist[N_CITIES * N_CITIES] = {
    0, 5, 7, 4, 6, 8, 5, 6, 2,
    3, 0, 6, 7, 5, 7, 5, 5, 7,
    9, 4, 0, 7, 6, 6, 4, 5, 7,
    6, 5, 3, 0, 8, 4, 6, 6, 5,
    9, 5, 7, 3, 0, 8, 5, 5, 9,
    7, 6, 5, 7, 4, 0, 5, 6, 7,
    5, 8, 5, 4, 6, 3, 0, 9, 5,
    5, 7, 4, 5, 4, 9, 1, 0, 7,
    6, 8, 5, 6, 5, 8, 4, 2, 0
};

/*
 * =====================================================================================================================
 * HAMILTONIAN
 * =====================================================================================================================
 */

/* Diagonal cost Hamiltonian over the full 2^N_QUBITS computational basis.
 *
 * Encoding: city (N_CITIES - 1) is fixed as the tour start and end.  The
 * remaining NUM_REG = N_CITIES - 1 cities are assigned to time slots via
 * qubit registers.  Register r occupies qubits [r*LEN_REG, (r+1)*LEN_REG)
 * and its unsigned integer value identifies the city visited at time slot r.
 *
 * Since 2^LEN_REG may exceed NUM_REG (e.g., LEN_REG=2 gives range [0,3] but
 * only cities [0,2] are valid for 4 cities), some register values are
 * out-of-range.  Additionally, valid register values may repeat, encoding a
 * city visited twice.  Both cases are infeasible.
 *
 * Feasible basis state: every register value is in [0, NUM_REG) and all
 * register values are distinct (a valid permutation of cities).
 *
 * ham[k] = tour cost for feasible k, 0 for infeasible k. */
static double ham[1 << N_QUBITS];

static void build_hamiltonian() {
    unsigned long long dim = 1ULL << N_QUBITS;
    unsigned long long mask = (1ULL << LEN_REG) - 1ULL;
    unsigned fixed = N_CITIES - 1;           /* city anchored at start and end of tour */

    for (unsigned long long k = 0; k < dim; ++k) {
        /* Decode the basis index into NUM_REG register values. */
        unsigned regs[NUM_REG];
        for (unsigned r = 0; r < NUM_REG; ++r) {
            regs[r] = (unsigned)((k >> (r * LEN_REG)) & mask);
        }

        /* Feasibility check.
         * 'seen' is a bitmask: bit j is set if city j has already appeared
         * in an earlier register, detecting duplicate city assignments. */
        unsigned seen = 0;
        int feasible = 1;
        for (unsigned r = 0; r < NUM_REG; ++r) {
            if (regs[r] >= NUM_REG) {        /* out-of-range: register value exceeds valid city count */
                feasible = 0; break;
            }
            if (seen & (1u << regs[r])) {    /* duplicate: city already visited in an earlier time slot */
                feasible = 0; break;
            }
            seen |= (1u << regs[r]);
        }
        if (!feasible) { ham[k] = 0.0; continue; }

        /* Tour cost: fixed -> regs[0] -> regs[1] -> ... -> regs[last] -> fixed.
         * Indices into dist[] are directional (row = origin, col = destination),
         * so this is correct for asymmetric distance matrices. */
        double cost = dist[fixed * N_CITIES + regs[0]];
        for (unsigned r = 0; r + 1 < NUM_REG; ++r) {
            cost += dist[regs[r] * N_CITIES + regs[r + 1]];
        }
        cost += dist[regs[NUM_REG - 1] * N_CITIES + fixed];
        ham[k] = cost;
    }
}

/*
 * =====================================================================================================================
 * STATE
 * =====================================================================================================================
 */

static state_t iota = {
    .type   = PURE,
    .data   = NULL,
    .qubits = N_QUBITS
};

/* Precomputed basis index for the identity tour: register r holds value r,
 * encoding the tour (fixed -> 0 -> 1 -> ... -> NUM_REG-1 -> fixed). */
static unsigned long long identity_idx;

static void compute_identity_idx() {
    identity_idx = 0;
    for (unsigned r = 0; r < NUM_REG; ++r) {
        identity_idx |= ((unsigned long long)r) << (r * LEN_REG);
    }
}

static void init_identity_tour() {
    unsigned long long dim = 1ULL << N_QUBITS;
    cplx_t *data = calloc(dim, sizeof *data);
    data[identity_idx] = 1.0 + 0.0 * I;
    state_init(&iota, N_QUBITS, &data);
}

/*
 * =====================================================================================================================
 * OPERATORS
 * =====================================================================================================================
 */

/* A precomputed swap operator: n_pairs disjoint qubit-SWAP pairs defining
 * a composite permutation U.  Used with composite_exp to apply exp(-i·θ·U). */
typedef struct {
    unsigned  n_pairs;
    qubit_t  *qa;
    qubit_t  *qb;
} swap_pairs_t;

static unsigned     n_bin_ops;
static swap_pairs_t bin_ops[N_BIN_OPS_UPPER];
static qubit_t      bin_qa[N_BIN_PAIRS_UPPER];
static qubit_t      bin_qb[N_BIN_PAIRS_UPPER];

/* Intermediate representation: register-level transpositions before expansion
 * to qubit-level SWAP pairs. */
typedef struct {
    unsigned n_trans;
    unsigned a[NUM_REG];                     /* 0-based register indices */
    unsigned b[NUM_REG];
} reg_trans_t;

static unsigned ceil_log2_u(unsigned x) {
    unsigned r = 0, v = 1;
    while (v < x) { v <<= 1; ++r; }
    return r == 0 ? 1u : r;
}

/* Generate the binary insertion sequence Xi(NUM_REG) from Lemma 6 of
 * Schwietering et al., "Exhaustive and feasible parametrisation with
 * applications to the travelling salesperson problem."
 *
 * The recursion builds Xi(m) from Xi(m-1) for m = 2..NUM_REG:
 *   1. raise(Xi(m-1)): shift all register indices by +1
 *   2. Append ceil(log2(m)) new operators pi_ell, each consisting of
 *      min{m - 2^(ell-1), 2^(ell-1)} disjoint transpositions (j, j + 2^(ell-1))
 *      for j = 1..min{...}.  These are "binary stride" permutations that can
 *      move element 1 to any position in [m] via binary addressing.
 *
 * Unlike adjacency and bubbles, some binsert operators contain multiple
 * disjoint transpositions sharing a single parameter.  For example, with 5
 * cities (NUM_REG=4), pi_2 for m=4 produces {(1,3),(2,4)} — two register
 * transpositions exponentiated jointly in one composite_exp call. */
static void build_bin_ops() {
    reg_trans_t seq[N_BIN_OPS_UPPER];
    unsigned sz = 0;

    for (unsigned m = 2; m <= NUM_REG; ++m) {
        /* Raise: shift all existing register indices by +1. */
        for (unsigned x = 0; x < sz; ++x) {
            for (unsigned y = 0; y < seq[x].n_trans; ++y) {
                seq[x].a[y] += 1;
                seq[x].b[y] += 1;
            }
        }
        /* Append pi_ell for ell = 1..ceil(log2(m)). */
        unsigned lmax = ceil_log2_u(m);
        for (unsigned ell = 1; ell <= lmax; ++ell) {
            unsigned step = 1u << (ell - 1);
            unsigned n_trans = (m - step < step) ? m - step : step;
            if (n_trans == 0) continue;
            seq[sz].n_trans = n_trans;
            for (unsigned k = 1; k <= n_trans; ++k) {
                seq[sz].a[k - 1] = k;           /* 1-based */
                seq[sz].b[k - 1] = k + step;
            }
            ++sz;
        }
    }

    /* Convert 1-based register indices to 0-based. */
    for (unsigned x = 0; x < sz; ++x) {
        for (unsigned y = 0; y < seq[x].n_trans; ++y) {
            seq[x].a[y] -= 1;
            seq[x].b[y] -= 1;
        }
    }

    /* Expand register transpositions to qubit-level SWAP pairs. */
    unsigned pair_off = 0;
    for (unsigned x = 0; x < sz; ++x) {
        unsigned base = pair_off;
        for (unsigned y = 0; y < seq[x].n_trans; ++y) {
            unsigned r1 = seq[x].a[y], r2 = seq[x].b[y];
            for (unsigned b = 0; b < LEN_REG; ++b) {
                bin_qa[pair_off + b] = (qubit_t)(r1 * LEN_REG + b);
                bin_qb[pair_off + b] = (qubit_t)(r2 * LEN_REG + b);
            }
            pair_off += LEN_REG;
        }
        bin_ops[x].n_pairs = seq[x].n_trans * LEN_REG;
        bin_ops[x].qa      = &bin_qa[base];
        bin_ops[x].qb      = &bin_qb[base];
    }

    n_bin_ops = sz;
}

/* Apply exp(-i * theta * U) where U is a product of n_pairs disjoint SWAP gates.
 *
 * Since U is a product of disjoint SWAPs, U^2 = I (involutory), so:
 *   exp(-i * theta * U) |psi> = cos(theta) |psi> - i * sin(theta) * U |psi>
 *
 * This requires a full state copy because the disjoint SWAPs are exponentiated
 * jointly, not individually. */
static void composite_exp(state_t *state, double theta,
                          const qubit_t *qa, const qubit_t *qb, unsigned n_pairs) {
    double cs = cos(theta);
    double sn = sin(theta);
    state_t phi = state_cp(state);
    for (unsigned i = 0; i < n_pairs; ++i) {
        swap_gate(&phi, qa[i], qb[i]);
    }
    unsigned long long dim = 1ULL << state->qubits;
    for (unsigned long long k = 0; k < dim; ++k) {
        cplx_t a  = state_get(state, (idx_t)k, 0);
        cplx_t ua = state_get(&phi,  (idx_t)k, 0);
        state_set(state, (idx_t)k, 0, cs * a - I * sn * ua);
    }
    state_free(&phi);
}

/* Binsert mixer: each precomputed operator is applied with its own parameter. */
static void mixer(state_t *state, const double *params) {
    for (unsigned i = 0; i < n_bin_ops; ++i) {
        composite_exp(state, params[i], bin_ops[i].qa, bin_ops[i].qb, bin_ops[i].n_pairs);
    }
}

/*
 * =====================================================================================================================
 * OBJECTIVE FUNCTION
 * =====================================================================================================================
 */

static double obj(unsigned short n, const double *params, void *userdata) {
    (void)n;
    (void)userdata;

    init_identity_tour();

    for (unsigned i = 0; i < DEPTH; ++i) {
        mixer(&iota, &params[i * n_bin_ops]);
    }

    double val = mean(&iota, ham);
    state_free(&iota);
    return val;
}

/*
 * =====================================================================================================================
 * MAIN
 * =====================================================================================================================
 */

int main() {
    build_hamiltonian();
    compute_identity_idx();
    build_bin_ops();

    unsigned n_par = n_bin_ops * DEPTH;

    double *par = malloc(n_par * sizeof *par);
    for (unsigned i = 0; i < n_par; ++i) {
        par[i] = M_PI / 4.0;
    }

    opt_result_t *r = NULL;
    opt_status_t s = opt_run(OPT_NLOPT_COBYLA, (unsigned short)n_par, par, obj, NULL, NULL, NULL, &r);

    if (s == OPT_OK) {
        printf("Optimal tour cost:\t%.1f\n", OPTIMUM);
        printf("Approximation ratio:\t%.4f\n", r->value / OPTIMUM);
        printf("Optimal parameters:\t[");
        for (unsigned i = 0; i < n_par; ++i) {
            printf("%.4f%s", par[i], i + 1 < n_par ? ", " : "");
        }
        printf("]\n");
        printf("\nHistory:\n");
        for (unsigned i = 0; i < r->n_calls; ++i) {
            printf("%-8d:%8.4f\n", r->history[i].call_idx, r->history[i].obj_value / OPTIMUM);
        }
    }

    free(par);
    opt_free(r);
    return 0;
}
