#include "Grid.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <functional>

#if __cplusplus >= 201703L && __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#endif
#ifdef LIKWID_PERFMON
    #include <likwid.h>
#endif

Grid::Grid(int columns_, int rows_)
    : columns(columns_ + 2 * HALO), rows(rows_ + 2 * HALO) {
    for (int i = 0; i < 4; ++i) ghost[i] = Dirichlet;
    arrayPtr = new double[rows * columns];
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows * columns; i++)
        arrayPtr[i] = 0.0;
}

Grid::Grid(int columns_, int rows_, BC_TYPE *ghost_)
    : columns(columns_ + 2 * HALO), rows(rows_ + 2 * HALO) {
    for (int i = 0; i < 4; ++i) ghost[i] = ghost_[i];
    arrayPtr = new double[rows * columns];
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows * columns; i++)
        arrayPtr[i] = 0.0;
}

Grid::Grid(const Grid &s)
{
    if (this != &s) {
        rows = s.rows;
        columns = s.columns;
        for (int i = 0; i < 4; ++i) ghost[i] = s.ghost[i];
        int totGrids = rows * columns;
        arrayPtr = new double[totGrids];
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < totGrids; ++i)
            arrayPtr[i] = s.arrayPtr[i];
    }
}

bool Grid::readFile(const std::string& name, bool halo)
{
    int file_rows, file_columns;
    std::ifstream file(name, std::ios::in);
    if (file.is_open()) {
        file >> file_rows >> file_columns;
        assert((file_rows == numGrids_y(halo)) && (file_columns == numGrids_x(halo)));
        int shift = halo ? 0 : HALO;
        for (int i = shift; i < numGrids_y(true) - shift; ++i)
            for (int j = shift; j < numGrids_x(true) - shift; ++j)
                file >> (*this)(i, j);
    }
    file.close();
    return false;
}

bool Grid::writeFile(const std::string& name, bool halo)
{
    std::ofstream file(name, std::ios::out);
    if (file.is_open()) {
        file << numGrids_y(halo) << " " << numGrids_x(halo) << "\n";
        int shift = halo ? 0 : HALO;
        for (int i = shift; i < numGrids_y(true) - shift; ++i) {
            for (int j = shift; j < numGrids_x(true) - shift; ++j)
                file << (*this)(i, j) << "\t";
            file << "\n";
        }
    }
    file.close();
    return false;
}

void Grid::print(bool halo)
{
    int shift = halo ? 0 : HALO;
    for (int i = shift; i < numGrids_y(true) - shift; ++i) {
        for (int j = shift; j < numGrids_x(true) - shift; ++j)
            printf(" %10.5f", (*this)(i, j));
        printf("\n");
    }
}

int Grid::numGrids_x(bool halo) const
{
    int halo_x = halo ? 0 : 2 * HALO;
    return (columns - halo_x);
}

int Grid::numGrids_y(bool halo) const
{
    int halo_y = halo ? 0 : 2 * HALO;
    return (rows - halo_y);
}

int Grid::numGrids(bool halo) const
{
    return numGrids_y(halo) * numGrids_x(halo);
}

void Grid::fillBoundary(std::function<double(int, int)> func, Direction dir)
{
    int nx = numGrids_x(true), ny = numGrids_y(true);
    if (dir == WEST)
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ny; ++j)
            (*this)(j, 0) = func(0, j);
    if (dir == EAST)
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ny; ++j)
            (*this)(j, nx - 1) = func(nx - 1, j);
    if (dir == NORTH)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nx; ++i)
            (*this)(ny - 1, i) = func(i, ny - 1);
    if (dir == SOUTH)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nx; ++i)
            (*this)(0, i) = func(i, 0);
}

void Grid::fill(double val, bool halo)
{
    int shift = halo ? 0 : HALO;
    #pragma omp parallel for schedule(static)
    for (int j = shift; j < numGrids_y(true) - shift; ++j) {
        for (int i = shift; i < numGrids_x(true) - shift; ++i)
            (*this)(j, i) = val;
    }
}

void Grid::rand(bool halo, unsigned int seed)
{
    int shift = halo ? 0 : HALO;
    #pragma omp parallel for schedule(static)
    for (int j = shift; j < numGrids_y(true) - shift; ++j) {
        for (int i = shift; i < numGrids_x(true) - shift; ++i)
            (*this)(j, i) = rand_r(&seed) / static_cast<double>(RAND_MAX);
    }
}

void Grid::fill(std::function<double(int, int)> func, bool halo)
{
    int shift = halo ? 0 : HALO;
    #pragma omp parallel for schedule(static)
    for (int j = shift; j < numGrids_y(true) - shift; ++j) {
        for (int i = shift; i < numGrids_x(true) - shift; ++i)
            (*this)(j, i) = func(i, j);
    }
}

void Grid::copyToHalo(std::function<double(int, int)> shift_func, Direction dir)
{
    int nx = numGrids_x(true), ny = numGrids_y(true);
    if (dir == WEST)
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ny; ++j)
            (*this)(j, 0) = (*this)(j, HALO) + shift_func(0, j);
    if (dir == EAST)
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < ny; ++j)
            (*this)(j, nx - 1) = (*this)(j, nx - 1 - HALO) + shift_func(nx, j);
    if (dir == NORTH)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nx; ++i)
            (*this)(ny - 1, i) = (*this)(ny - 1 - HALO, i) + shift_func(i, ny);
    if (dir == SOUTH)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nx; ++i)
            (*this)(0, i) = (*this)(HALO, i) + shift_func(i, 0);
}

void Grid::swap(Grid &other)
{
    assert(rows == other.rows && columns == other.columns);
    double *temp = (*this).arrayPtr;
    (*this).arrayPtr = other.arrayPtr;
    other.arrayPtr = temp;
}

Grid::~Grid()
{
    delete[] arrayPtr;
    arrayPtr = NULL;
}

// AXPBY: lhs[:] = a*x[:] + b*y[:]
void axpby(Grid *lhs, double a, Grid *x, double b, Grid *y, bool halo)
{
    START_TIMER(AXPBY);
#ifdef DEBUG
    assert((lhs->numGrids_y(true) == x->numGrids_y(true)) && (lhs->numGrids_x(true) == x->numGrids_x(true)));
    assert((y->numGrids_y(true) == x->numGrids_y(true)) && (y->numGrids_x(true) == x->numGrids_x(true)));
#endif
    int shift = halo ? 0 : HALO;
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("AXPBY");
#endif
    #pragma omp parallel for schedule(static)
    for (int j = shift; j < lhs->numGrids_y(true) - shift; ++j) {
        #pragma omp simd
        for (int i = shift; i < lhs->numGrids_x(true) - shift; ++i)
            (*lhs)(j, i) = (a * (*x)(j, i)) + (b * (*y)(j, i));
    }
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("AXPBY");
#endif
    STOP_TIMER(AXPBY);
}

// COPY: lhs[:] = a*rhs[:]
void copy(Grid *lhs, double a, Grid *rhs, bool halo)
{
    START_TIMER(COPY);
#ifdef DEBUG
    assert((lhs->numGrids_y(true) == rhs->numGrids_y(true)) && (lhs->numGrids_x(true) == rhs->numGrids_x(true)));
#endif
    int shift = halo ? 0 : HALO;
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("COPY");
#endif
    #pragma omp parallel for schedule(static)
    for (int j = shift; j < lhs->numGrids_y(true) - shift; ++j) {
        #pragma omp simd
        for (int i = shift; i < lhs->numGrids_x(true) - shift; ++i)
            (*lhs)(j, i) = a * (*rhs)(j, i);
    }
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("COPY");
#endif
    STOP_TIMER(COPY);
}

// DOT PRODUCT
double dotProduct(Grid* __restrict__ x, Grid* __restrict__ y, bool halo)
{
    START_TIMER(DOT_PRODUCT);
#ifdef DEBUG
    assert((y->numGrids_y(true) == x->numGrids_y(true)) && (y->numGrids_x(true) == x->numGrids_x(true)));
#endif
    const int xSize = x->numGrids_x(true);
    const int ySize = x->numGrids_y(true);
    const int shift = halo ? 0 : HALO;
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("DOT_PRODUCT");
#endif
    double dot_res = 0.0;
    #pragma omp parallel for reduction(+:dot_res) schedule(static)
    for (int j = shift; j < ySize - shift; ++j) {
        #pragma omp simd reduction(+:dot_res)
        for (int i = shift; i < xSize - shift; ++i) {
            int idx = j * xSize + i;
            dot_res = std::fma(x->arrayPtr[idx], y->arrayPtr[idx], dot_res);
        }
    }
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("DOT_PRODUCT");
#endif
    STOP_TIMER(DOT_PRODUCT);
    return dot_res;
}

bool isSymmetric(Grid *u, double tol, bool halo)
{
    bool flag = true;
    const int xSize = u->numGrids_x(true);
    const int ySize = u->numGrids_y(true);
    int shift = halo ? 0 : HALO;
    #pragma omp parallel for schedule(static)
    for (int j = shift; j < ySize - shift; ++j) {
        for (int i = shift; i < xSize - shift; ++i) {
            if (std::abs((*u)(i, j) - (*u)(j, i)) > tol)
                flag = false;
        }
    }
    return flag;
}

// NON-CLASS MEMBER FUNCTIONS:

bool writeGnuplotFile(const std::string& name, Grid &src, double len_x, double len_y, bool halo)
{
    std::cout << "Writing solution file..." << std::endl;
    fs::path file_path = name;
    fs::path dir = file_path.parent_path();
    if (!dir.empty() && !fs::exists(dir)) {
        if (fs::create_directories(dir)) {
            std::cout << "Directory created: " << dir << '\n';
        } else {
            std::cout << "Failed to create directory: " << dir << '\n';
        }
    }
    std::ofstream file(name, std::ios::out);
    double hx = len_x / static_cast<double>(src.numGrids_x(true) - 1.0);
    double hy = len_y / static_cast<double>(src.numGrids_y(true) - 1.0);
    int shift = halo ? 0 : HALO;
    if (file.is_open()) {
        file << "#" << "x" << "\t" << "y" << "\t" << "u(x,y)" << "\n";
        for (int yIndex = shift; yIndex < (src.numGrids_y(true) - shift); yIndex++)
            for (int xIndex = shift; xIndex < (src.numGrids_x(true) - shift); xIndex++)
                file << xIndex * hx << "\t" << yIndex * hy << "\t" << src(yIndex, xIndex) << "\n";
    }
    file.close();
    return true;
}
