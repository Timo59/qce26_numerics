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
#define DEPTH     ((N_CITIES - 1) / 2)
#define OPTIMUM   25.0
#define N_PAR     (2 * DEPTH)                /* 1 mixer param + 1 separator param per layer */

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
 * a composite permutation U. Used with composite_exp to apply exp(-i·θ·U). */
typedef struct {
    unsigned  n_pairs;
    qubit_t  *qa;
    qubit_t  *qb;
} swap_pairs_t;

/* Adjacency: NUM_REG - 1 operators, one per adjacent register pair. */
#define N_ADJ_OPS  (NUM_REG - 1)
static swap_pairs_t adj_ops[N_ADJ_OPS];
static qubit_t   adj_qa[N_ADJ_OPS * LEN_REG];
static qubit_t   adj_qb[N_ADJ_OPS * LEN_REG];

/* Precompute swap pairs for adjacent register transpositions (r, r+1). */
static void build_adj_ops() {
    for (unsigned r = 0; r < N_ADJ_OPS; ++r) {
        unsigned base = r * LEN_REG;
        for (unsigned b = 0; b < LEN_REG; ++b) {
            adj_qa[base + b] = (qubit_t)(r * LEN_REG + b);
            adj_qb[base + b] = (qubit_t)((r + 1) * LEN_REG + b);
        }
        adj_ops[r].n_pairs = LEN_REG;
        adj_ops[r].qa      = &adj_qa[base];
        adj_ops[r].qb      = &adj_qb[base];
    }
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

/* Adjacency mixer: apply exp(-i * theta * SWAP(r, r+1)) sequentially for each
 * adjacent register pair.  All NUM_REG - 1 transpositions share the same
 * parameter theta. */
static void mixer(state_t *state, const double theta) {
    for (unsigned i = 0; i < N_ADJ_OPS; ++i) {
        composite_exp(state, theta, adj_ops[i].qa, adj_ops[i].qb, adj_ops[i].n_pairs);
    }
}

/* Phase separator: exp(-i * gamma * H_C) applied via diagonal exponentiation. */
static void separator(state_t *state, const double gamma) {
    exp_diag(state, ham, gamma);
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
        mixer(&iota, params[2 * i]);
        separator(&iota, params[2 * i + 1]);
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
    build_adj_ops();

    double par[N_PAR];
    for (unsigned i = 0; i < N_PAR; ++i) {
        par[i] = M_PI / 4.0;
    }

    opt_result_t *r = NULL;
    opt_status_t s = opt_run(OPT_NLOPT_COBYLA, N_PAR, par, obj, NULL, NULL, NULL, &r);

    if (s == OPT_OK) {
        printf("Optimal tour cost:\t%.1f\n", OPTIMUM);
        printf("Approximation ratio:\t%.4f\n", r->value / OPTIMUM);
        printf("Optimal parameters:\t[");
        for (unsigned i = 0; i < N_PAR; ++i) {
            printf("%.4f%s", par[i], i + 1 < N_PAR ? ", " : "");
        }
        printf("]\n");
        printf("\nHistory:\n");
        for (unsigned i = 0; i < r->n_calls; ++i) {
            printf("%-8d:%8.4f\n", r->history[i].call_idx, r->history[i].obj_value / OPTIMUM);
        }
    }

    opt_free(r);
    return 0;
}
