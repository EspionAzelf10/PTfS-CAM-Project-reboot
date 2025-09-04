// Solver.cpp - with SIMD, FMA, cache blocking, and correct OpenMP usage
#include "Solver.h"
#include "Grid.h"
#include <cmath>
#include <omp.h>

#define IS_VALID(a) (!(std::isnan(a) || std::isinf(a)))

// Cache blocking parameters (tunable)
#define BLOCK_Y 32
#define BLOCK_X 64

SolverClass::SolverClass(PDE *pde_, Grid *x_, Grid *b_) : pde(pde_), x(x_), b(b_) {}

// ---------------------------
// Conjugate Gradient (CG)
// ---------------------------
int SolverClass::CG(int niter, double tol)
{
    Grid *p = new Grid(pde->numGrids_x(), pde->numGrids_y());
    Grid *v = new Grid(pde->numGrids_x(), pde->numGrids_y());

    int iter = 0;
    double lambda = 0.0;
    double alpha_0 = 0.0, alpha_1 = 0.0;

    // Calculate residual: p = b - A*x
    pde->applyStencil(p, x);

    {
        const int xSize = p->numGrids_x(true);
        const int ySize = p->numGrids_y(true);
        const int shift = HALO;
        const int start = shift * xSize;
        const int end = (ySize - shift) * xSize;

        double sum = 0.0;
        #pragma omp parallel reduction(+:sum)
        {
            double * __restrict__ pptr = p->arrayPtr;
            double * __restrict__ bptr = b->arrayPtr;

            #pragma omp for schedule(static)
            for (int idx = start; idx < end; ++idx) {
                double val = bptr[idx] - pptr[idx];
                pptr[idx] = val;
                sum = std::fma(val, val, sum);
            }
        }
        alpha_0 = sum;
    }

    Grid *r = new Grid(*p); // r = p

    START_TIMER(CG);

    while ((iter < niter) && (alpha_0 > tol * tol) && (IS_VALID(alpha_0)))
    {
        // v = A * p
        pde->applyStencil(v, p);

        double denom = dotProduct(v, p);
        lambda = alpha_0 / denom;

        // x = x + lambda * p
        axpby(x, 1.0, x, lambda, p);

        // ---- r update + norm (loop fusion) ----
        {
            const int xSize = r->numGrids_x(true);
            const int ySize = r->numGrids_y(true);
            const int shift = HALO;
            const int x_start = shift, x_end = xSize - shift;
            const int y_start = shift, y_end = ySize - shift;

            double global_sum = 0.0;
            #pragma omp parallel reduction(+:global_sum)
            {
                double * __restrict__ rptr = r->arrayPtr;
                double * __restrict__ vptr = v->arrayPtr;

                #pragma omp for schedule(static)
                for (int by = y_start; by < y_end; by += BLOCK_Y) {
                    int by_max = (by + BLOCK_Y > y_end) ? y_end : by + BLOCK_Y;

                    for (int row = by; row < by_max; ++row) {
                        int base = row * xSize;
                        #pragma omp simd reduction(+:global_sum)
                        for (int ix = x_start; ix < x_end; ++ix) {
                            int idx = base + ix;
                            double rv = rptr[idx] - lambda * vptr[idx];
                            rptr[idx] = rv;
                            global_sum = std::fma(rv, rv, global_sum);
                        }
                    }
                }
            }
            alpha_1 = global_sum;
        }
        // ---------------------------------------

        // p = r + (alpha_1/alpha_0) * p
        axpby(p, 1.0, r, alpha_1 / alpha_0, p);

        alpha_0 = alpha_1;

#ifdef DEBUG
        printf("iter = %d, res = %.15e\n", iter, alpha_0);
#endif
        ++iter;
    }

    STOP_TIMER(CG);

    if (!IS_VALID(alpha_0)) {
        printf("\x1B[31mWARNING: NaN/INF detected after iteration %d\x1B[0m\n", iter);
    }

    delete p;
    delete v;
    delete r;

    return iter;
}

// ---------------------------
// Preconditioned CG (PCG)
// ---------------------------
int SolverClass::PCG(int niter, double tol)
{
    Grid* r = new Grid(pde->numGrids_x(), pde->numGrids_y());
    Grid* z = new Grid(pde->numGrids_x(), pde->numGrids_y());
    Grid* v = new Grid(pde->numGrids_x(), pde->numGrids_y());

    int iter = 0;
    double lambda = 0.0;
    double alpha_0 = 0.0, alpha_1 = 0.0;
    double res_norm_sq = 0.0;

    // r = b - A*x
    pde->applyStencil(r, x);

    {
        const int xSize = r->numGrids_x(true);
        const int ySize = r->numGrids_y(true);
        const int shift = HALO;
        const int start = shift * xSize;
        const int end = (ySize - shift) * xSize;

        double sum = 0.0;
        #pragma omp parallel reduction(+:sum)
        {
            double * __restrict__ rptr = r->arrayPtr;
            double * __restrict__ bptr = b->arrayPtr;

            #pragma omp for schedule(static)
            for (int idx = start; idx < end; ++idx) {
                double rv = bptr[idx] - rptr[idx];
                rptr[idx] = rv;
                sum = std::fma(rv, rv, sum);
            }
        }
        res_norm_sq = sum;
    }

    pde->GSPreCon(r, z);

    alpha_0 = dotProduct(r, z);
    Grid* p = new Grid(*z);

    START_TIMER(PCG);

    while ((iter < niter) && (res_norm_sq > tol * tol) && (IS_VALID(res_norm_sq)))
    {
        // v = A * p
        pde->applyStencil(v, p);

        double denom = dotProduct(v, p);
        lambda = alpha_0 / denom;

        // x = x + lambda * p
        axpby(x, 1.0, x, lambda, p);

        // ---- r update + norm (loop fusion) ----
        {
            const int xSize = r->numGrids_x(true);
            const int ySize = r->numGrids_y(true);
            const int shift = HALO;
            const int x_start = shift, x_end = xSize - shift;
            const int y_start = shift, y_end = ySize - shift;

            double global_sum = 0.0;
            #pragma omp parallel reduction(+:global_sum)
            {
                double * __restrict__ rptr = r->arrayPtr;
                double * __restrict__ vptr = v->arrayPtr;

                #pragma omp for schedule(static)
                for (int by = y_start; by < y_end; by += BLOCK_Y) {
                    int by_max = (by + BLOCK_Y > y_end) ? y_end : by + BLOCK_Y;

                    for (int row = by; row < by_max; ++row) {
                        int base = row * xSize;
                        #pragma omp simd reduction(+:global_sum)
                        for (int ix = x_start; ix < x_end; ++ix) {
                            int idx = base + ix;
                            double rv = rptr[idx] - lambda * vptr[idx];
                            rptr[idx] = rv;
                            global_sum = std::fma(rv, rv, global_sum);
                        }
                    }
                }
            }
            res_norm_sq = global_sum;
        }
        // ---------------------------------------

        pde->GSPreCon(r, z);

        alpha_1 = dotProduct(r, z);

        // p = z + (alpha_1/alpha_0) * p
        axpby(p, 1.0, z, alpha_1 / alpha_0, p);

        alpha_0 = alpha_1;

#ifdef DEBUG
        printf("iter = %d, res = %.15e\n", iter, res_norm_sq);
#endif
        ++iter;
    }

    STOP_TIMER(PCG);

    if (!IS_VALID(res_norm_sq)) {
        printf("\x1B[31mWARNING: NaN/INF detected after iteration %d\x1B[0m\n", iter);
    }

    delete r;
    delete z;
    delete v;
    delete p;

    return iter;
}
