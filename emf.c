/* EM algorithm for vanilla Gaussian mixture model. */

#include "emf.h"
#include <math.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>

/* Should diagonal or full covariance matrices be used? */
#ifdef DIAGONAL
#define dchoose(a,b) a
#else
#define dchoose(a,b) b
#endif

typedef dchoose(sdiag,stwod) smat;
typedef dchoose(diag,twod) mat;

struct gaussian {
  spoint mean;			/* center of gaussian */
  smat covar;			/* covariance matrix */
  double d;			/* determinant of covariance matrix */
};

struct cluster {
  struct gaussian g;
  double prior;
};

void zero_mat(mat m)
{
  dchoose(zero_diag,zero_twod)(m);
}

void scale_mat(mat m, double s)
{
  dchoose(scale_diag, scale_twod)(m,s);
}

double det(mat m)
{
  return dchoose(det_diag,det_twod)(m);
}

void add_outer_prod(point p, mat m, double scale)
{
  dchoose(diag_add_outer_prod, twod_add_outer_prod)(p, m, scale);
}

double inner_prod_inv(point x, mat m)
{
  return dchoose(diag_inner_prod_inv, twod_inner_prod_inv)(x, m);
}

/* add two log numbers */
double log_plus(double x, double y)
{
  if (x<y) { double t=x; x=y; y=t; }
  return log1p(exp(y-x)) + x;
}

/* d-dimensional gaussian. */
double gaussian_pdf(point p, struct gaussian *g)
{
  spoint x;
  point_diff(p, g->mean, x);
  return exp(-inner_prod_inv(x, g->covar)/2)
    / sqrt(pow(2 * M_PI, D) * g->d);
}

/* d-dimensional gaussian. */
double log_gaussian_pdf(point p, struct gaussian *g)
{
  spoint x;
  point_diff(p, g->mean, x);
  return -(inner_prod_inv(x,g->covar) + D*log(2*M_PI) + log(g->d))/2;
}

/* gaussian mixture pdf */
double mixture_pdf(point p, struct cluster *cs)
{
  int i;
  double total=0;
  FOR(i,K)
    total += cs[i].prior * gaussian_pdf(p, &cs[i].g);
  return total;
}

/* gaussian mixture pdf */
double log_mixture_pdf(point p, struct cluster *cs)
{
  int i;
  double log_total = log(cs[0].prior) + log_gaussian_pdf(p, &cs[0].g);
  for (i=1; i<K; i++)
    log_total = log_plus(log_total,
			 log(cs[i].prior) + log_gaussian_pdf(p, &cs[i].g));
  return log_total;
}

void accumulate_stats(point x, struct cluster *c, double w)
{
  add_outer_prod(x, c->g.covar, w);
  add_scaled_point(x, w, c->g.mean);
  c->prior += w;
}

void re_estimate(spoint *data, struct cluster *class)
{
  int i,k;
  struct cluster newclass[K];

  FOR(k,K)			/* zero stats */
    {
      zero_mat(newclass[k].g.covar);
      zero_point(newclass[k].g.mean);
      newclass[k].prior = 0;
    }

  FOR(i,N)
    {
      /* compute posterior of point in various classes. */
      double pr[K];

      FOR(k,K)
	pr[k] = class[k].prior * gaussian_pdf(data[i], &class[k].g);

      normalize_l1(pr, K);

      FOR(k,K)
	accumulate_stats(data[i], &newclass[k], pr[k]);
    }

  FOR(k,K)			/* adjust things */
    {
      scale_point(newclass[k].g.mean, 1/newclass[k].prior);
      scale_mat(newclass[k].g.covar, 1/newclass[k].prior);
      add_outer_prod(newclass[k].g.mean, newclass[k].g.covar, -1);
      newclass[k].g.d = det(newclass[k].g.covar);
      newclass[k].prior /= N;
    }

  FOR(k,K)
    class[k] = newclass[k];
}

int skip_to_eol(FILE *fd)
{
  int c;
  do {
    c = getc(fd);
  } while (c != EOF && c != '\n');
  return c;
}

/* skip whitespace and comments, which are ; to end of line */
void skip_whitespace(FILE *fd)
{
  int c;
  do {
    c = getc(fd);
    if (c==';')
      c = skip_to_eol(fd);
  } while (isspace(c));
  ungetc(c, fd);
}

double read_double(FILE *fd)
{
  double d;
  skip_whitespace(fd);
  if (fscanf(fd, "%lf", &d) != 1)
    {
      fprintf(stderr, "error: read_double.\n");
      exit(1);
    }
  return d;
}

void read_doubles(FILE *fd, double *p, int n)
{
  int i;
  for (i=0; i<n; i++)
    p[i] = read_double(fd);
}

void read_point(FILE *fd, point p)
{
  read_doubles(fd, p, D);
}

void read_twod(FILE *fd, twod m)
{
  read_doubles(fd, &m[0][0], D*D);
}

void read_diag(FILE *fd, diag m)
{
  read_point(fd, m);
}

void read_mat(FILE *fd, mat m)
{
  dchoose(read_diag, read_twod)(fd, m);
}

void read_data(FILE *fd, point data)
{
  read_doubles(fd, &data[0], D*N);
}

void read_gaussian(FILE *fd, struct gaussian *g)
{
  read_point(fd, g->mean);
  read_mat(fd, g->covar);
  g->d = det(g->covar);
}

void read_cluster(FILE *fd, struct cluster *c)
{
  /* cluster format: prior, gaussian */
  c->prior = read_double(fd);
  read_gaussian(fd, &c->g);
}

void read_clusters(FILE *fd, struct cluster *cs)
{
  int k;
  /* cluster format: prior, gaussian */
  for (k=0; k<K; k++)
    {
      cs[k].prior = read_double(fd);
      read_gaussian(fd, &cs[k].g);
    }
}

void print_point(FILE *fd, point p)
{
  int i;
  for (i=0; i<D; i++)
    fprintf(fd, " %G", p[i]);
  fprintf(fd, "\n");
}

void print_twod(FILE *fd, twod m)
{
  int i,j;
  for (i=0; i<D; i++)
    {
      fprintf(fd, "   ");
      for (j=0; j<D; j++)
	fprintf(fd, " %G", m[i][j]);
      fprintf(fd, "\n");
    }
  fprintf(fd, "\n");
}

void print_diag(FILE *fd, diag m)
{
  print_point(fd, m);
}

void print_mat(FILE *fd, mat m)
{
  dchoose(print_diag, print_twod)(fd, m);
}

void print_gaussian(FILE *fd, struct gaussian *g)
{
  fprintf(fd, "; mean\n");
  print_point(fd, g->mean);
  fprintf(fd, dchoose("; diagonal covariance matrix\n",
		      "; full covariance matrix\n"));
  print_mat(fd, g->covar);
}

void print_cluster(FILE *fd, struct cluster *c)
{
  fprintf(fd, "; prior\n");
  fprintf(fd, "%G\n", c->prior);
  print_gaussian(fd, &c->g);
}

void print_clusters(FILE *fd, struct cluster *cs, int epoch)
{
  int i;
  printf(";;; Epoch %d.\n", epoch);
  for (i=0; i<K; i++)
    {
      fprintf(fd, ";; Cluster %d\n", i);
      print_cluster(fd, &cs[i]);
    }
}

FILE *my_fopen(char *s, char *mode)
{
  FILE *fd;
  fd = fopen(s, mode);
  if (!fd) {
    fprintf(stderr, "Unable to open \"%s\" in mode \"%s\".\n", s, mode);
    exit(1);
  }
  return fd;
}

void label_dataset(struct cluster *cs, char *indata, int n, char *outdata)
{
  FILE *fd, *ofd;
  int i;
  spoint p;
  point pp = &p[0];

  fd = my_fopen(indata,"r");
  ofd = my_fopen(outdata,"w");

  for (i=0; i<n; i++)
    {
      read_point(fd, pp);
      fprintf(ofd, "%G\n", log_mixture_pdf(pp, cs));
    }
  fclose(fd);
  fclose(ofd);
}

void print_config(FILE *fd)
{
  fprintf(fd, "; %d datapoints, %d clusters, %d-dimensional.\n", N, K, D);
  fprintf(fd, dchoose("; Diagonal covariance matrices.\n",
		      "; Full covariance matrices.\n"));
}

int main(int argc, char **argv)
{
  int epochs;
  double d=0;
  int i;
  FILE *fd;
  spoint data[N];
  struct cluster class[K];

  if (argc != 4 && argc != 5 && argc != 7) {
    fprintf(stderr,
	    "Usage: %s epochs data clusters [labels [data n]].\n",
	    argv[0]);
    print_config(stderr);
    exit(2);
  }

  epochs = atoi(argv[1]);
  fd = my_fopen(argv[2],"r"); read_data(fd, data[0]); fclose(fd);
  fd = my_fopen(argv[3],"r"); read_clusters(fd, class); fclose(fd);

  print_config(stdout);

  FOR(i,K)
    d += class[i].prior;
  printf("; Initial sum of priors %G.\n", d);
  FOR(i,K)
    class[i].prior /= d;

  if (debug) print_clusters(stdout, class, 0);

  for(i=1; i<=epochs; i++) {
    re_estimate(data, class);
    if (debug) print_clusters(stdout, class, i);
  }
  if (!debug) print_clusters(stdout, class, epochs);

  if (argc != 4)
    label_dataset(class,
		  ( argc>5 ? argv[5] : argv[2] ),
		  ( argc>5 ? atoi(argv[6]) : N ),
		  argv[4]);
  return 0;
}
