#include <cstdlib>
#include <iostream>
#include <ostream>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#define REP(i, start, end) for (uint32_t i = start; i < end; ++i)

struct Config {
    uint32_t const nx;
    uint32_t const ny;
    uint32_t const nt;
    uint32_t const nit;
    double const dx;
    double const dy;
    double const dt;
    double const rho;
    double const nu;

    Config(
        uint32_t nx,
        uint32_t ny,
        uint32_t nt,
        uint32_t nit,
        double dt,
        double rho,
        double nu
    ):
        nx(nx),
        ny(ny),
        nt(nt),
        nit(nit),
        dx(2. / ((double)nx - 1.)),
        dy(2. / ((double)ny - 1.)),
        dt(dt),
        rho(rho),
        nu(nu)
    {}
};

template<typename T>
class Matrix {
    T * const elems; // [row0, row1, ...]
    uint32_t const num_row;
    uint32_t const num_col;
public:

    Matrix &operator=(Matrix const &m) {
        assert(num_row == m.num_row);
        assert(num_col == m.num_col);
        memcpy(elems, m.elems, num_row * num_col * sizeof(T));
        return *this;
    }

    ~Matrix() {
        free(elems);
    }

    Matrix(uint32_t num_row, uint32_t num_col)
      : num_col(num_col), num_row(num_row), elems((T*)malloc(num_row * num_col * sizeof(T)))
    {
        memset(elems, 0, num_row * num_col * sizeof(T));
    }

    inline T &operator()(uint32_t row, uint32_t col) {
        return elems[col + row * num_col];
    }

    inline T const &operator()(uint32_t row, uint32_t col)  const {
        return elems[col + row * num_col];
    }

    inline void copy_col(uint32_t to, uint32_t from) {
        #pragma omp parallel for schedule(static)
        REP(r, 0, num_row) {
            (*this)(r, to) = (*this)(r, from);
        }
    }

    inline void copy_row(uint32_t to, uint32_t from) {
        memcpy(&elems[to * num_col], &elems[from * num_col], num_col * sizeof(T));
    }

    inline void fill_col(uint32_t col, T val) {
        #pragma omp parallel for schedule(static)
        REP(r, 0, num_row) {
            (*this)(r, col) = val;
        }
    }

    inline void fill_row(uint32_t row, T val) {
        #pragma omp parallel for schedule(static)
        REP(c, 0, num_col) {
            (*this)(row, c) = val;
        }
    }

    T const *raw() const {
        return this->elems;
    }

    friend std::ostream &operator<<(std::ostream &o, Matrix<T> const &m) {
        REP(r, 0, m.num_row) {
            REP(c, 0, m.num_col) {
                o << m(r, c) << ", ";
            }
            o << '\n';
        }
        return o;
    }
};

template<>
inline void Matrix<double>::fill_row(uint32_t row, double val) {
    uint32_t const STROK = (256 / 8) / sizeof(double);
    __m256d vals = _mm256_set1_pd(val);

    auto num_iter = num_col / STROK;
    auto *row_begin = &elems[num_col * row];
    for (int i = 0; i < num_col / STROK; ++i) {
        _mm256_storeu_pd(row_begin + i * STROK, vals);
    }
    _mm256_storeu_pd(row_begin + num_col - STROK, vals);
}

inline static double sq(double a) {
    return a * a;
}

class Simulator {
    Config const c;
    uint32_t current_step;
    Matrix<double> b;
    Matrix<double> u;
    Matrix<double> v;
    Matrix<double> p;

public:
    Simulator(Config c) :
        c(c),
        u(Matrix<double>(c.nx, c.ny)),
        v(Matrix<double>(c.nx, c.ny)),
        p(Matrix<double>(c.nx, c.ny)),
        b(Matrix<double>(c.nx, c.ny)),
        current_step(0)
    {}

    Matrix<double> const& get_u() { return u; }
    Matrix<double> const& get_v() { return v; }
    Matrix<double> const& get_p() { return p; }

    void update() {
        // リアロケートのコストをなくすため static を使う。
        static Matrix<double> pn(c.nx, c.ny);
        static Matrix<double> un(c.nx, c.ny);
        static Matrix<double> vn(c.nx, c.ny);

        if (current_step >= c.nt) return;

        #pragma omp parallel for schedule(static)
        REP(j, 1, c.ny - 1) {
            REP(i, 1, c.nx - 1) {
                b(j, i) = c.rho * (1 / c.dt *
                        ((u(j, i+1) - u(j, i-1)) / (2 * c.dx) + (v(j+1, i) - v(j-1, i)) / (2 * c.dy)) -
                        sq((u(j, i+1) - u(j, i-1)) / (2 * c.dx)) - 2 * ((u(j+1, i) - u(j-1, i)) / (2 * c.dy) *
                         (v(j, i+1) - v(j, i-1)) / (2 * c.dx)) - sq((v(j+1, i) - v(j-1, i)) / (2 * c.dy)));
            }
        }

        REP(it, 0, c.nit) {
            pn = p;
            #pragma omp parallel for schedule(static)
            REP(j, 1, c.ny - 1) {
                REP(i, 1, c.nx - 1) {
                    p(j, i) = (sq(c.dy) * (pn(j, i+1) + pn(j, i-1)) +
                                  sq(c.dx) * (pn(j+1, i) + pn(j-1, i)) -
                                  b(j, i) * sq(c.dx) * sq(c.dy))
                                 / (2 * (sq(c.dx) + sq(c.dy)));
                }
            }
            p.copy_col(c.nx - 1, c.nx - 2);
            p.copy_col(0, 1);
            p.copy_row(0, 1);
            p.fill_row(c.ny - 1, 0);
        }

        un = u;
        vn = v;

        #pragma omp parallel for schedule(static)
        REP(j, 1, c.ny - 1) {
            REP(i, 1, c.nx - 1) {
                u(j, i) = un(j, i) - un(j, i) * c.dt / c.dx * (un(j, i) - un(j, i - 1))
                                   - un(j, i) * c.dt / c.dy * (un(j, i) - un(j - 1, i))
                                   - c.dt / (2 * c.rho * c.dx) * (p(j, i+1) - p(j, i-1))
                                   + c.nu * c.dt / sq(c.dx) * (un(j, i+1) - 2 * un(j, i) + un(j, i-1))
                                   + c.nu * c.dt / sq(c.dy) * (un(j+1, i) - 2 * un(j, i) + un(j-1, i));

                v(j, i) = vn(j, i) - vn(j, i) * c.dt / c.dx * (vn(j, i) - vn(j, i - 1))
                                   - vn(j, i) * c.dt / c.dy * (vn(j, i) - vn(j - 1, i))
                                   - c.dt / (2 * c.rho * c.dx) * (p(j+1, i) - p(j-1, i))
                                   + c.nu * c.dt / sq(c.dx) * (vn(j, i+1) - 2 * vn(j, i) + vn(j, i-1))
                                   + c.nu * c.dt / sq(c.dy) * (vn(j+1, i) - 2 * vn(j, i) + vn(j-1, i));
            }
        }

        #pragma omp parallel
        {
            #pragma omp sections
            {
                #pragma omp section
                { v.fill_col(0, 0); }
                #pragma omp section
                { v.fill_col(c.nx - 1, 0); }
                #pragma omp section
                { v.fill_row(0, 0); }
                #pragma omp section
                { v.fill_row(c.ny - 1, 0); }

                #pragma omp section
                { u.fill_col(0, 0); }
                #pragma omp section
                { u.fill_col(c.nx - 1, 0); }
                #pragma omp section
                { u.fill_row(0, 0); }
                #pragma omp section
                { u.fill_row(c.ny - 1, 1); }
            }
        }
        u(c.ny - 1, 0) = 1;
        u(c.ny - 1, c.nx - 1) = 1;

        ++current_step;
    }
};

extern "C" {
    Config *conf_new(
        uint32_t nx,
        uint32_t ny,
        uint32_t nt,
        uint32_t nit,
        double dt,
        double rho,
        double nu
    ) {
        return new Config(nx, ny, nt, nit, dt, rho, nu);
    }

    Simulator *simu_new(Config *conf) {
        return new Simulator(*conf);
    }

    void simu_update(Simulator *simu) {
        simu->update();
    }

    double const *simu_get_u(Simulator *simu) {
        return simu->get_u().raw();
    }

    double const *simu_get_v(Simulator *simu) {
        return simu->get_v().raw();
    }

    double const *simu_get_p(Simulator *simu) {
        return simu->get_p().raw();
    }
}
