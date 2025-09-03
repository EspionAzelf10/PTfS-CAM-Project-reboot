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
void PDE::applyStencil(Grid* lhs, Grid* x)
{
    START_TIMER(APPLY_STENCIL);

#ifdef DEBUG
    assert((lhs->numGrids_y(true)==grids_y) && (lhs->numGrids_x(true)==grids_x));
    assert((x->numGrids_y(true)==grids_y) && (x->numGrids_x(true)==grids_x));
#endif
    const int xSize = numGrids_x(true);
    const int ySize = numGrids_y(true);

    const double w_x = 1.0/(h_x*h_x);
    const double w_y = 1.0/(h_y*h_y);
    const double w_c = 2.0*w_x + 2.0*w_y;

    // Direct array access for better vectorization
    double* __restrict__ lhsPtr = lhs->arrayPtr;
    double* __restrict__ xPtr = x->arrayPtr;

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("APPLY_STENCIL");
#endif

    // Cache blocking for better locality
    const int BLOCK_SIZE = 64;  // Adjusted for L1 cache size

    #pragma omp parallel for collapse(2) schedule(static) proc_bind(close)
    for(int jb = 1; jb < ySize-1; jb += BLOCK_SIZE) {
        for(int ib = 1; ib < xSize-1; ib += BLOCK_SIZE) {
            const int jmax = std::min(jb + BLOCK_SIZE, ySize-1);
            const int imax = std::min(ib + BLOCK_SIZE, xSize-1);

            for(int j = jb; j < jmax; ++j) {
                const int row = j * xSize;
                const int rowUp = (j-1) * xSize;
                const int rowDown = (j+1) * xSize;

                #pragma omp simd aligned(lhsPtr,xPtr:64)
                for(int i = ib; i < imax; ++i) {
                    lhsPtr[row + i] = w_c * xPtr[row + i] 
                                   - w_y * (xPtr[rowDown + i] + xPtr[rowUp + i])
                                   - w_x * (xPtr[row + i + 1] + xPtr[row + i - 1]);
                }
            }
        }
    }

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("APPLY_STENCIL");
#endif

    STOP_TIMER(APPLY_STENCIL);
}

void PDE::GSPreCon(Grid* rhs, Grid *x)
{
    START_TIMER(GS_PRE_CON);

#ifdef DEBUG
    assert((rhs->numGrids_y(true)==grids_y) && (rhs->numGrids_x(true)==grids_x));
    assert((x->numGrids_y(true)==grids_y) && (x->numGrids_x(true)==grids_x));
#endif
    const int xSize = x->numGrids_x(true);
    const int ySize = x->numGrids_y(true);

    const double w_x = 1.0/(h_x*h_x);
    const double w_y = 1.0/(h_y*h_y);
    const double w_c = 1.0/(2.0*w_x + 2.0*w_y);

    double* __restrict__ xPtr = x->arrayPtr;
    const double* __restrict__ rhsPtr = rhs->arrayPtr;

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("GS_PRE_CON");
#endif

    // Forward sweep - must remain sequential but can use SIMD within rows
    for(int j = 1; j < ySize-1; ++j) {
        const int row = j * xSize;
        const int rowUp = (j-1) * xSize;

        #pragma omp simd aligned(xPtr,rhsPtr:64)
        for(int i = 1; i < xSize-1; ++i) {
            const int idx = row + i;
            xPtr[idx] = w_c * (rhsPtr[idx] + 
                              w_y * xPtr[rowUp + i] + 
                              w_x * xPtr[idx - 1]);
        }
    }

    // Backward sweep
    for(int j = ySize-2; j > 0; --j) {
        const int row = j * xSize;
        const int rowDown = (j+1) * xSize;

        #pragma omp simd aligned(xPtr:64)
        for(int i = xSize-2; i > 0; --i) {
            const int idx = row + i;
            xPtr[idx] += w_c * (w_y * xPtr[rowDown + i] + 
                               w_x * xPtr[idx + 1]);
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
