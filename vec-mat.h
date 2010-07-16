#ifndef D
#define D 2			/* dimensionality */
#endif

typedef double *point;
typedef double (*twod)[D];
typedef double *diag;

extern void zero_point(point x);
extern void zero_twod(twod m);
extern void zero_diag(diag m);
extern void point_diff(point x, point y, point z);
extern void add_scaled_point(point x, double s, point y);
extern void scale_point(point x, double s);
extern void scale_twod(twod m, double s);
extern void scale_diag(diag d, double s);
extern double dot(point x, point y);
extern void matmult(twod m, point x, point y);
extern void diagmult(diag d, point x, point y);
extern void identity_twod(twod l);
extern void identity_diag(diag d);
extern void copy_twod(twod m, twod u);
extern void copy_diag(diag m, diag u);
extern void lu_decompose(twod m, twod l, twod u);
extern double det_twod(twod m);
extern double det_diag(diag m);
extern void solve_linear_system(int n, double (*mm)[n], double *b, double *x);
extern void solve_diag_system(diag m, point b, point x);
extern void twod_add_outer_prod(point p, twod m, double scale);
extern void diag_add_outer_prod(point p, diag m, double scale);
extern double inner_prod(point p, twod m);
extern double twod_inner_prod_inv(point x, twod m);
extern double diag_inner_prod_inv(point x, diag m);

typedef double spoint[D];	/* vector */
typedef double stwod[D][D];	/* matrix */
typedef double sdiag[D];	/* diagonal matrix */

#define FOR(i,n) for((i)=0; (i)<(n); (i)++)
