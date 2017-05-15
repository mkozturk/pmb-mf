#ifndef COMMON_H 
#define COMMON_H

#define max(a,b)				\
  ({ __typeof__ (a)_a = (a);			\
    __typeof__ (b)_b = (b);			\
    _a > _b ? _a : _b; })

#define min(a,b)				\
  ({ __typeof__ (a)_a = (a);			\
    __typeof__ (b)_b = (b);			\
    _a < _b ? _a : _b; })

#ifdef SP
#define opt_prec_t float
#endif

#ifdef DP
#define opt_prec_t double
#endif

typedef struct {
  int exit;
  opt_prec_t fval;
  opt_prec_t gradient_norm;
  int iterations;
  int evaluations;
} Output;

typedef struct {
  int maxiter;
  int maxinneriter;
  opt_prec_t tol;
  int M;
} Options;

//#define TIMER
//#define DEBUG

#endif
