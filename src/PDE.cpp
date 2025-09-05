//back to normal
#include "PDE.h"
#include <math.h>
#include <iostream>
#include <omp.h>

#ifdef LIKWID_PERFMON
    #include <likwid.h>
#endif
//default boundary function as in ex01
double defaultBoundary(int i, int j, double h_x, double h_y)
{
    return sin(M_PI*i*h_x)*sinh(M_PI*j*h_y);
}
//default rhs function as in ex01
double zeroFunc(int i, int j, double h_x, double h_y)
{
    return 0 + 0*i*h_x + 0*j*h_y;
}

//Constructor
PDE::PDE(int len_x_, int len_y_, int grids_x_, int grids_y_):len_x(len_x_), len_y(len_y_), grids_x(grids_x_+2*HALO), grids_y(grids_y_+2*HALO)
{
    h_x = static_cast<double>(len_x)/(grids_x-1.0);
    h_y = static_cast<double>(len_y)/(grids_y-1.0);

    initFunc = zeroFunc;

    //by default all boundary is Dirichlet
    for (int i=0; i<4; ++i)
        boundary[i] = Dirichlet;

    for (int i=0; i<4; ++i)
        boundaryFunc[i] = zeroFunc;
}

int PDE::numGrids_x(bool halo)
{
    int halo_x = halo ? 0:2*HALO;
    return (grids_x-halo_x);
}

int PDE::numGrids_y(bool halo)
{
    int halo_y = halo ? 0:2*HALO;
    return (grids_y-halo_y);
}


void PDE::init(Grid *grid)
{
#ifdef DEBUG
    assert((grid->numGrids_y(true)==grids_y) && (grid->numGrids_x(true)==grids_x));
#endif
    grid->fill(std::bind(initFunc,std::placeholders::_1,std::placeholders::_2,h_x,h_y));
}

// Boundary Condition
void PDE::applyBoundary(Grid *u)
{
#ifdef DEBUG
    assert((u->numGrids_y(true)==grids_y) && (u->numGrids_x(true)==grids_x));
#endif
    if(boundary[NORTH]==Dirichlet){
        u->fillBoundary(std::bind(boundaryFunc[NORTH],std::placeholders::_1,std::placeholders::_2,h_x,h_y),NORTH);
    }
    if(boundary[SOUTH]==Dirichlet){
        u->fillBoundary(std::bind(boundaryFunc[SOUTH],std::placeholders::_1,std::placeholders::_2,h_x,h_y),SOUTH);
    }
    if(boundary[EAST]==Dirichlet){
        u->fillBoundary(std::bind(boundaryFunc[EAST],std::placeholders::_1,std::placeholders::_2,h_x,h_y),EAST);
    }
    if(boundary[WEST]==Dirichlet){
        u->fillBoundary(std::bind(boundaryFunc[WEST],std::placeholders::_1,std::placeholders::_2,h_x,h_y),WEST);
    }
}

//It refreshes Neumann boundary, 2 nd argument is to allow for refreshing with 0 shifts, ie in coarser levels
void PDE::refreshBoundary(Grid *u)
{
#ifdef DEBUG
    assert((u->numGrids_y(true)==grids_y) && (u->numGrids_x(true)==grids_x));
#endif
    if(boundary[NORTH]==Neumann){
        u->copyToHalo(std::bind(boundaryFunc[NORTH],std::placeholders::_1,std::placeholders::_2,h_x,h_y),NORTH);
    }
    if(boundary[SOUTH]==Neumann){
        u->copyToHalo(std::bind(boundaryFunc[SOUTH],std::placeholders::_1,std::placeholders::_2,h_x,h_y),SOUTH);
    }
    if(boundary[EAST]==Neumann){
        u->copyToHalo(std::bind(boundaryFunc[EAST],std::placeholders::_1,std::placeholders::_2,h_x,h_y),EAST);
    }
    if(boundary[WEST]==Neumann){
        u->copyToHalo(std::bind(boundaryFunc[WEST],std::placeholders::_1,std::placeholders::_2,h_x,h_y),WEST);
    }
}


//Applies stencil operation on to x
//i.e., lhs = A*x
// inside PDE.cpp (replace existing PDE::applyStencil implementation)
void PDE::applyStencil(Grid* lhs, Grid* x)
{
    START_TIMER(APPLY_STENCIL);

#ifdef DEBUG
    assert((lhs->numGrids_y(true)==grids_y) && (lhs->numGrids_x(true)==grids_x));
    assert((x->numGrids_y(true)==grids_y) && (x->numGrids_x(true)==grids_x));
#endif

    const int xSize = numGrids_x(true);
    const int ySize = numGrids_y(true);

    // weights for 5-point finite-difference Laplacian / Poisson-like operator
    const double w_x = 1.0/(h_x*h_x);
    const double w_y = 1.0/(h_y*h_y);
    const double w_c = 2.0*w_x + 2.0*w_y;

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("APPLY_STENCIL");
#endif

    // Tunable blocking parameter (rows per block). Adjust to your cache.
#ifndef STENCIL_BLOCK_Y
    const int BLOCK_Y = 32;
#else
    const int BLOCK_Y = STENCIL_BLOCK_Y;
#endif

    // Raw pointers for faster indexing (same layout used elsewhere)
    double * __restrict__ lhsPtr = lhs->arrayPtr;
    double * __restrict__ xPtr   = x->arrayPtr;

    // We keep the same loop bounds as your original code: 1 .. size-2
    const int y_start = 1;
    const int y_end   = ySize - 1; // exclusive upper bound in blocked loop logic below
    const int x_start = 1;
    const int x_end   = xSize - 1;

    // Parallelize across row-blocks. Each thread works on whole inner x ranges
    #pragma omp parallel
    {
        // Each thread executes chunks of blocks
        #pragma omp for schedule(static)
        for (int by = y_start; by < y_end; by += BLOCK_Y) {
            int by_max = by + BLOCK_Y;
            if (by_max > y_end) by_max = y_end;

            // Process rows in the block
            for (int j = by; j < by_max; ++j) {
                int base = j * xSize;

                // Inner loop — vectorizeable
                #pragma omp simd
                for (int i = x_start; i < x_end; ++i) {
                    int idx = base + i;
                    // neighbor indices (linear)
                    // x(j,i)     -> xPtr[idx]
                    // x(j+1,i)   -> xPtr[idx + xSize]
                    // x(j-1,i)   -> xPtr[idx - xSize]
                    // x(j,i+1)   -> xPtr[idx + 1]
                    // x(j,i-1)   -> xPtr[idx - 1]
                    lhsPtr[idx] = w_c * xPtr[idx]
                                - w_y * (xPtr[idx + xSize] + xPtr[idx - xSize])
                                - w_x * (xPtr[idx + 1]     + xPtr[idx - 1]);
                }
            }
        }
    } // end parallel

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("APPLY_STENCIL");
#endif

    STOP_TIMER(APPLY_STENCIL);
}


void PDE::GSPreCon(Grid* rhs, Grid *x)
{
    START_TIMER(GS_PRE_CON);

#ifdef DEBUG
    assert((rhs->numGrids_y(true) == grids_y) && (rhs->numGrids_x(true) == grids_x));
    assert((x->numGrids_y(true)   == grids_y) && (x->numGrids_x(true)   == grids_x));
#endif

    const int xSize = x->numGrids_x(true);
    const int ySize = x->numGrids_y(true);

    const double w_x = 1.0/(h_x*h_x);
    const double w_y = 1.0/(h_y*h_y);
    const double w_c = 1.0/(2.0*w_x + 2.0*w_y);

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("GS_PRE_CON");
#endif

    double* __restrict__ xPtr = x->arrayPtr;
    double* __restrict__ rhsPtr = rhs->arrayPtr;

    const int sStart = 2;
    const int sEnd   = (xSize - 2) + (ySize - 2);

    #pragma omp parallel
    {
        // Forward wavefront
        for (int s = sStart; s <= sEnd; ++s) {
            int i_low  = std::max(1, s - (ySize - 2));
            int i_high = std::min(xSize - 2, s - 1);

            #pragma omp for schedule(static)
            for (int i = i_low; i <= i_high; ++i) {
                int j = s - i;
                int idx = j * xSize + i;

                // Prefetch next row data for cache
                __builtin_prefetch(xPtr + idx + xSize, 0, 3);
                __builtin_prefetch(rhsPtr + idx + xSize, 0, 3);

                // SIMD vectorize this loop in caller with #pragma omp simd
                xPtr[idx] = w_c * (rhsPtr[idx]
                                 + w_y * xPtr[idx - xSize]
                                 + w_x * xPtr[idx - 1]);
            }
        }

        #pragma omp barrier

        // Backward wavefront
        for (int s = sEnd; s >= sStart; --s) {
            int i_low  = std::max(1, s - (ySize - 2));
            int i_high = std::min(xSize - 2, s - 1);

            #pragma omp for schedule(static)
            for (int i = i_low; i <= i_high; ++i) {
                int j = s - i;
                int idx = j * xSize + i;

                __builtin_prefetch(xPtr + idx - xSize, 0, 3);

                xPtr[idx] += w_c * (w_y * xPtr[idx + xSize]
                                  + w_x * xPtr[idx + 1]);
            }
        }
    }

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("GS_PRE_CON");
#endif

    STOP_TIMER(GS_PRE_CON);
}




int PDE::solve(Grid *x, Grid *b, Solver type, int niter, double tol)
{
    SolverClass solver(this, x, b);
    if(type==CG)
    {
        return solver.CG(niter, tol);
    }
    else if(type==PCG)
    {
        return solver.PCG(niter, tol);
    }
    else
    {
        printf("Solver not existing\n");
        return -1;
    }
}
