#include "vec-mat.h"

/* square a double */
inline double sqr(double x)
{
  return x*x;
}

/* zero out vector */
void zero_point(point x)
{
  int i;
  FOR(i,D)
    x[i] = 0;
}

/* zero out matrix */
void zero_twod(twod m)
{
  int i,j;
  FOR(i,D)
    FOR(j,D)
      m[i][j] = 0;
}

void zero_diag(diag m)
{
  zero_point(m);
}

/* vector difference */
void point_diff(point x, point y, point z)
{
  int i;
  FOR(i,D)
    z[i] = x[i] - y[i];
}

/* add scaled point to target point */
void add_scaled_point(point x, double s, point y)
{
  int i;
  FOR(i,D)
    y[i] += s*x[i];
}

/* scale a vector */
void scale_point(point x, double s)
{
  int i;
  FOR(i,D)
    x[i] *= s;
}

/* scale a matrix */
void scale_twod(twod m, double s)
{
  int i,j;
  FOR(i,D)
    FOR(j,D)
      m[i][j] *= s;
}

/* scale a diagonal matrix */
void scale_diag(diag d, double s)
{
  scale_point(d, s);
}

/* vector dot product */
double dot(point x, point y)
{
  double r=0;
  int i;
  FOR(i,D)
    r += x[i] * y[i];
  return r;
}

/* Matrix multiplication by a vector */
void matmult(twod m, point x, point y)
{
  int i,j;
  zero_point(y);
  FOR(i,D)
    FOR(j,D)
      y[i] += m[i][j] * x[j];
}

/* Diagonal matrix mult by a vector */
void diagmult(diag d, point x, point y)
{
  int i;
  FOR(i,D)
    y[i] = d[i]*x[i];
}

/* Set matrix to identity. */
void identity_twod(twod l)
{
  int i;
  zero_twod(l);
  FOR(i,D)
    l[i][i] = 1;
}

void identity_diag(diag d)
{
  int i;
  FOR(i,D)
    d[i] = 1;
}

/* Copy matrix m into u */
void copy_twod(twod m, twod u)
{
  int i, j;
  FOR(i,D)
    FOR(j,D)
      u[i][j] = m[i][j];
}

void copy_diag(diag m, diag u)
{
  int i;
  FOR(i,D)
    u[i] = m[i];
}

/* Decompose m=l.u, where l is lower triangular and u is upper triangular.
   The matrix l ends up with only 1's along its diagonal.
   Note: this approach is wasteful of storage. */
void lu_decompose(twod m, twod l, twod u)
{
  int i, j;
  /* set l to identity */
  identity_twod(l);
  /* set u = m */
  copy_twod(m, u);
  /* apply row operations to zero out sub-diagonal elements of u. */
  /* apply same row ops to l. */
  for (i=0; i<D-1; i++) {
    double a = u[i][i];
    for (j=i+1; j<D; j++) {
      double s = - u[i][j] / a;
      add_scaled_point(u[i], s, u[j]);
      add_scaled_point(l[i], s, l[j]);
    }
  }
}

/* Determinant */
double det_twod(twod m)
{
  switch (D)
    {
    case 1:
      return m[0][0];
    case 2:
      return m[0][0]*m[1][1] - m[0][1]*m[1][0];
    default:
      {
	stwod l, u;
	int i;
	double d=1;
	lu_decompose(m,l,u);
	FOR(i,D)
	  d *= l[i][i] * u[i][i];
	return d;
      }
    }
}

double det_diag(diag m)
{
  switch (D)
    {
    case 1:
      return m[0];
    case 2:
      return m[0]*m[1];
    case 3:
      return m[0]*m[1]*m[2];
    case 4:
      return m[0]*m[1]*m[2]*m[3];
    default:
      {
	double d=1;
	int i;
	FOR(i,D)
	  d *= m[i];
	return d;
      }
    }
}

/* This solves the matrix equation Mx=b for x (M=mm, an n by n
   matrix).  Only x is modified. */

void solve_linear_system(int n, double (*mm)[n], double *b, double *x)
{
  int i,j,k;
  double a;
  double m[n][n];		/* working copy of input, will side effect */

  FOR(i,n)
    FOR(j,n)
      m[i][j] = mm[i][j];

  FOR(i,n)
    x[i] = b[i];

  /* Forward pass: */

  for (i=0; i<n; i++) {
    a = 1/m[i][i];
    /* multiply row i by a: */
    for (j=i+1; j<n; j++)
      m[i][j] *= a;
    x[i] *= a;

    for (k=i+1; k<n; k++) {
      a = -m[k][i];
      /* add a times row i to row k */
      for (j=i+1; j<n; j++)
	m[k][j] += m[i][j] * a;
      x[k] += x[i] * a;
    }
  }

  /* Backward pass: */

  for (i=n-1; i>=0; i--)
    for (k=i-1; k>=0; k--) {
      a = -m[k][i];
      /* add a times row i to row k */
      for (j=i+1; j<n; j++)
	m[k][j] += m[i][j] * a;
      x[k] += x[i] * a;
    }
}

/* Solve Mx=b for x when M is diagonal */
void solve_diag_system(diag m, point b, point x)
{
  int i;
  FOR(i,D)
    x[i] = b[i] / m[i];
}

/* outer product */
void twod_add_outer_prod(point p, twod m, double scale)
{
  int i,j;
  FOR(i,D)
    FOR(j,D)
      m[i][j] += scale * p[i] * p[j];
}

void diag_add_outer_prod(point p, diag m, double scale)
{
  int i;
  FOR(i,D)
    m[i] += scale * sqr(p[i]);
}

/* inner product */
double inner_prod(point p, twod m)
{
  int i,j;
  double r=0;
  FOR(i,D)
    FOR(j,D)
      r += p[i] * m[i][j] * p[j];
  return r;
}

/* inner product using inverse of matrix */
double twod_inner_prod_inv(point x, twod m)
{
  spoint y;
  solve_linear_system(D, m, x, y);
  return dot(x, y);
}

/* inner product using inverse of diagonal matrix */
double diag_inner_prod_inv(point x, diag m)
{
  /* Gosh is this silly: */
  spoint y;
  solve_diag_system(m, x, y);
  return dot(x,y);
}

/* Normalize (so they sum to 1, i.e., unit L1 norm) the elements of a vector. */
/* Returns previous sum. */
double normalize_l1(double *p, int k)
{
  int i;
  double total=0;
  FOR(i,k)
    total += p[i];
  FOR(i,k)
    p[i] /= total;
  return total;
}
