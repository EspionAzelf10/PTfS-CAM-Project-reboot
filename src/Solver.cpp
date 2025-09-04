// Solver.cpp - simple loop fusion only (no kernel fusion)
#include "Solver.h"
#include "Grid.h"
#include <cmath>
#include <omp.h>

#define IS_VALID(a) (!(std::isnan(a) || std::isinf(a)))

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

    // Calculate residual
    // p = A*x (stencil)
    pde->applyStencil(p, x);

    // Fuse: p = b - p  (axpby)  AND alpha_0 = dot(p,p)
    {
        const int xSize = p->numGrids_x(true);
        const int ySize = p->numGrids_y(true);
        const int shift = HALO; // same convention as other kernels (halo omitted)
        const int start = shift * xSize;
        const int end = (ySize - shift) * xSize;

        double sum = 0.0;
        // Parallel fused loop: write p and accumulate norm
        #pragma omp parallel for reduction(+:sum)
        for (int idx = start; idx < end; ++idx) {
            double val = (*b).arrayPtr[idx] - p->arrayPtr[idx];  // b - A*x
            p->arrayPtr[idx] = val;
            sum += val * val;
        }
        alpha_0 = sum;
    }

    Grid *r = new Grid(*p); // r = p

    START_TIMER(CG);

    while ((iter < niter) && (alpha_0 > tol * tol) && (IS_VALID(alpha_0)))
    {
        // v = A * p
        pde->applyStencil(v, p);

        // denom = <v, p>
        double denom = dotProduct(v, p);
        lambda = alpha_0 / denom;

        // x = x + lambda * p (keep kernel)
        axpby(x, 1.0, x, lambda, p);

        // ---- Simple loop fusion: update r and compute alpha_1 = dot(r,r) in one pass ----
        {
            const int xSize = r->numGrids_x(true);
            const int ySize = r->numGrids_y(true);
            const int shift = HALO;
            const int start = shift * xSize;
            const int end = (ySize - shift) * xSize;

            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (int idx = start; idx < end; ++idx) {
                double rv = r->arrayPtr[idx] - lambda * v->arrayPtr[idx]; // r = r - lambda*v
                r->arrayPtr[idx] = rv;
                sum += rv * rv;
            }
            alpha_1 = sum;
        }
        // ------------------------------------------------------------------------------------

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

    // Initial residual: r = A*x
    pde->applyStencil(r, x);

    // Fuse: r = b - r  (axpby) AND res_norm_sq = dot(r,r)
    {
        const int xSize = r->numGrids_x(true);
        const int ySize = r->numGrids_y(true);
        const int shift = HALO;
        const int start = shift * xSize;
        const int end = (ySize - shift) * xSize;

        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int idx = start; idx < end; ++idx) {
            double rv = b->arrayPtr[idx] - r->arrayPtr[idx];
            r->arrayPtr[idx] = rv;
            sum += rv * rv;
        }
        res_norm_sq = sum;
    }

    // Preconditioner z = M^{-1} r
    pde->GSPreCon(r, z);

    alpha_0 = dotProduct(r, z);
    Grid* p = new Grid(*z);

    START_TIMER(PCG);

    while ((iter < niter) && (res_norm_sq > tol * tol) && (IS_VALID(res_norm_sq)))
    {
        // v = A * p
        pde->applyStencil(v, p);

        // denom = <v, p>
        double denom = dotProduct(v, p);
        lambda = alpha_0 / denom;

        // x = x + lambda * p
        axpby(x, 1.0, x, lambda, p);

        // ---- Simple loop fusion: r = r - lambda*v  and compute res_norm_sq = dot(r,r) ----
        {
            const int xSize = r->numGrids_x(true);
            const int ySize = r->numGrids_y(true);
            const int shift = HALO;
            const int start = shift * xSize;
            const int end = (ySize - shift) * xSize;

            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum)
            for (int idx = start; idx < end; ++idx) {
                double rv = r->arrayPtr[idx] - lambda * v->arrayPtr[idx];
                r->arrayPtr[idx] = rv;
                sum += rv * rv;
            }
            res_norm_sq = sum;
        }
        // ------------------------------------------------------------------------------------

        // z = M^{-1} r
        pde->GSPreCon(r, z);

        // alpha_1 = <r, z>
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
