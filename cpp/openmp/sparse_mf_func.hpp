#ifndef SMF_F_H
#define SMF_F_H

#include "common.h"
#include <iostream>

#ifdef CPUV3
#include <immintrin.h>
#endif

using namespace std;

opt_prec_t scfac;
opt_prec_t* div_term;

//#define LDIM 20
int LDIM = 20; //default value for LDIM

#define data_coord_t int 
#define data_point_t int

#ifdef SP
#define data_val_t float
#endif
#ifdef DP
#define data_val_t double
#endif

double divTerm_Z1Time = 0.0f;
double Z2Time = 0.0f;

struct SparseMatrix {
public:
  SparseMatrix(data_coord_t _nrows, data_coord_t _ncols, data_point_t _nnnz) : 
    nrows(_nrows), ncols(_ncols), nnnz(_nnnz), n((LDIM * nrows) + (LDIM * ncols)) {

    crs_ptrs = new data_point_t[nrows + 1];
    crs_colids = new data_coord_t[nnnz];
    crs_values = new data_val_t[nnnz];
    
    ccs_ptrs = new data_point_t[ncols + 1];
    ccs_rowids = new data_coord_t[nnnz];
    ccs_translator = new data_point_t[nnnz];
  }

  ~SparseMatrix() {
    delete[] crs_ptrs;
    delete[] crs_colids;
    delete[] crs_values;

    delete[] ccs_ptrs;
    delete[] ccs_rowids;
    delete[] ccs_translator;
  }
  
  data_coord_t nrows;
  data_coord_t ncols;
  data_point_t nnnz;
  int n;
  
  data_point_t* crs_ptrs;
  data_coord_t* crs_colids;
    
  data_point_t* ccs_ptrs;
  data_coord_t* ccs_rowids;
  data_point_t* ccs_translator;

  data_val_t* crs_values;
};

#ifdef CPUV1
opt_prec_t dTerm_Z1update(SparseMatrix* mat, opt_prec_t* Z1, opt_prec_t* Z1update, opt_prec_t* Z2) {
  opt_prec_t scfac = ((double)(1.0f)) / mat->nnnz;

  opt_prec_t totalcost = 0;
#pragma omp parallel
  {
#pragma omp for schedule(runtime) reduction(+:totalcost)
    for (data_coord_t i = 0; i < mat->nrows; i++){
      opt_prec_t *myZ1 = Z1 + (i * LDIM);
      opt_prec_t *myZ1U = Z1update + (i * LDIM);
      
      memset(myZ1U, 0, sizeof(opt_prec_t) * LDIM);

      data_point_t start = mat->crs_ptrs[i];
      data_point_t end = mat->crs_ptrs[i+1];

      for(data_point_t ptr = start; ptr < end; ptr++) {

	opt_prec_t *myZ2  = Z2 + (mat->crs_colids[ptr] * LDIM);

	opt_prec_t diff = -(mat->crs_values[ptr]);
	for (int k = 0; k < LDIM; k++) {
	  diff += myZ1[k] * myZ2[k];
	}
	div_term[ptr] = diff;	
	totalcost += (diff * diff);

	opt_prec_t coef =  diff * scfac;
	for (int k = 0; k < LDIM; k++) {
	  myZ1U[k] += myZ2[k] * coef;
	}
      }
    }
  }
  return totalcost * 0.5 * scfac;
}

void Z2update(SparseMatrix* mat, opt_prec_t* Z1, 
	      opt_prec_t* Z2update) {
  opt_prec_t scfac = ((double)(1.0f)) / mat->nnnz;

#pragma omp parallel for schedule(runtime)
  for (data_coord_t j = 0; j < mat->ncols; j++) {
    opt_prec_t *myZ2U = Z2update + (j * LDIM);
    memset (myZ2U, 0, sizeof(opt_prec_t) * LDIM);

    data_point_t start = mat->ccs_ptrs[j];
    data_point_t end = mat->ccs_ptrs[j + 1];
    for (data_point_t ptr = start; ptr < end; ptr++) {

      const opt_prec_t *myZ1 = Z1 + (mat->ccs_rowids[ptr] * LDIM);
      opt_prec_t coef = div_term[mat->ccs_translator[ptr]] * scfac;

      for (int k = 0; k < LDIM; k++) {
	myZ2U[k] += myZ1[k] * coef;
      }
    }
  }
}
#elif CPUV2
opt_prec_t dTerm_Z1update(SparseMatrix* mat, opt_prec_t* Z1, opt_prec_t* Z1update, opt_prec_t* Z2) {
  opt_prec_t scfac = ((double)(1.0f)) / mat->nnnz;

  opt_prec_t totalcost = 0;  
#pragma omp parallel
  {
    opt_prec_t diff_1, diff_2, diff_3, diff_4;
    opt_prec_t temp1, temp2, temp3, temp4;
    const opt_prec_t *myZ2_1, *myZ2_2, *myZ2_3, *myZ2_4;

#pragma omp for schedule(runtime) reduction(+:totalcost)
    for (data_coord_t i = 0; i < mat->nrows; i++) {
      const opt_prec_t *myZ1 = Z1 + (i * LDIM);
      opt_prec_t *myZ1U = Z1update + (i * LDIM);
      
      memset(myZ1U, 0, sizeof(opt_prec_t) * LDIM);
      
      data_point_t start = mat->crs_ptrs[i];
      data_point_t end = mat->crs_ptrs[i + 1];

      data_point_t ptr;      
      for (ptr = start; ptr < end - 3; ptr += 4) {
	myZ2_1  = Z2 + (mat->crs_colids[ptr] * LDIM);
	myZ2_2  = Z2 + (mat->crs_colids[ptr + 1] * LDIM);
	myZ2_3  = Z2 + (mat->crs_colids[ptr + 2] * LDIM);
	myZ2_4  = Z2 + (mat->crs_colids[ptr + 3] * LDIM);

	diff_1 = -(mat->crs_values[ptr]);
	diff_2 = -(mat->crs_values[ptr + 1]);
	diff_3 = -(mat->crs_values[ptr + 2]);
	diff_4 = -(mat->crs_values[ptr + 3]);

	int k = 0;
	for (; k < LDIM - 3; k += 4) {
	  temp1 = myZ1[k];
	  temp2 = myZ1[k+1];
	  temp3 = myZ1[k+2];
	  temp4 = myZ1[k+3];

	  diff_1 += (temp1 * myZ2_1[k] + temp2 * myZ2_1[k+1] + temp3 * myZ2_1[k+2] + temp4 * myZ2_1[k+3]);
	  diff_2 += (temp1 * myZ2_2[k] + temp2 * myZ2_2[k+1] + temp3 * myZ2_2[k+2] + temp4 * myZ2_2[k+3]);
	  diff_3 += (temp1 * myZ2_3[k] + temp2 * myZ2_3[k+1] + temp3 * myZ2_3[k+2] + temp4 * myZ2_3[k+3]);
	  diff_4 += (temp1 * myZ2_4[k] + temp2 * myZ2_4[k+1] + temp3 * myZ2_4[k+2] + temp4 * myZ2_4[k+3]);
	}

	for (; k < LDIM; k++) {
          opt_prec_t temp = myZ1[k];
          diff_1 += temp * myZ2_1[k];
          diff_2 += temp * myZ2_2[k];
          diff_3 += temp * myZ2_3[k];
          diff_4 += temp * myZ2_4[k];
	}

	div_term[ptr] = diff_1;
	div_term[ptr + 1] = diff_2;
	div_term[ptr + 2] = diff_3;
	div_term[ptr + 3] = diff_4;
	
	for (int k = 0; k < LDIM; k++) {
	  myZ1U[k] += (myZ2_1[k] * diff_1 +  myZ2_2[k] * diff_2 + myZ2_3[k] * diff_3 + myZ2_4[k] * diff_4) * scfac;
	}
	
	totalcost += diff_1 * diff_1 + diff_2 * diff_2 + diff_3 * diff_3 + diff_4 * diff_4;
      }
      
      for (; ptr < end; ptr++) {
	opt_prec_t *myZ2_1  = Z2 + (mat->crs_colids[ptr] * LDIM);
	
	diff_1 = -(mat->crs_values[ptr]);
	for (int k = 0; k < LDIM; k++) {
	  diff_1 += myZ1[k] * myZ2_1[k];
	}
	
	div_term[ptr] = diff_1;
	opt_prec_t coef = diff_1 * scfac;
	for (int k = 0; k < LDIM; k++) {
	  myZ1U[k] += myZ2_1[k] * coef;
	}
	totalcost += diff_1 * diff_1;
      }
    }
  }
  return totalcost * 0.5 * scfac;
}
void Z2update(SparseMatrix* mat, opt_prec_t* Z1, 
	      opt_prec_t* Z2update) {
  opt_prec_t scfac = ((double)(1.0f)) / mat->nnnz;

#pragma omp parallel for schedule(runtime)
  for (data_coord_t j = 0; j < mat->ncols; j++) {
    opt_prec_t *  myZ2U = Z2update + (j * LDIM);
    memset (myZ2U, 0, sizeof(opt_prec_t) * LDIM);
    
    data_point_t start = mat->ccs_ptrs[j];
    data_point_t end = mat->ccs_ptrs[j + 1];

    data_point_t ptr;
    for (ptr = start; ptr < end - 3; ptr += 4) {
      const opt_prec_t *myZ1_1  = Z1 + (mat->ccs_rowids[ptr] * LDIM);
      const opt_prec_t *myZ1_2  = Z1 + (mat->ccs_rowids[ptr + 1] * LDIM);
      const opt_prec_t *myZ1_3  = Z1 + (mat->ccs_rowids[ptr + 2] * LDIM);
      const opt_prec_t *myZ1_4  = Z1 + (mat->ccs_rowids[ptr + 3] * LDIM);
      
      const opt_prec_t cv_1 = div_term[mat->ccs_translator[ptr]];
      const opt_prec_t cv_2 = div_term[mat->ccs_translator[ptr + 1]];
      const opt_prec_t cv_3 = div_term[mat->ccs_translator[ptr + 2]];
      const opt_prec_t cv_4 = div_term[mat->ccs_translator[ptr + 3]];

      for (int k = 0; k < LDIM; k++) {
	myZ2U[k] += (myZ1_1[k] * cv_1 + myZ1_2[k] * cv_2 + 
		     myZ1_3[k] * cv_3 + myZ1_4[k] * cv_4) * scfac;
      }
    }

    for (; ptr < end; ptr++) {
      opt_prec_t *myZ1 = Z1 + (mat->ccs_rowids[ptr] * LDIM);
      
      opt_prec_t cv = div_term[mat->ccs_translator[ptr]] * scfac;
      for (int k = 0; k < LDIM; k++) {
	myZ2U[k] += myZ1[k] * cv;
      }
    }
  }
}
#elif CPUV3
opt_prec_t dTerm_Z1update(SparseMatrix* mat, opt_prec_t* Z1, opt_prec_t* Z1update, opt_prec_t* Z2) {
  opt_prec_t scfac = ((double)(1.0f)) / mat->nnnz;
  opt_prec_t totalcost = 0;
  
#pragma omp parallel
  {
    const opt_prec_t *myZ2_1, *myZ2_2, *myZ2_3, *myZ2_4;
    opt_prec_t diff_1;
    
    __m256d diffs; 
    __m256d z2s;

    double* z2s0 = ((double*)&z2s);
    double* z2s2 = ((double*)&z2s) + 2;

    double* diffs0 = ((double*)&diffs);
    double* diffs2 = ((double*)&diffs) + 2;

#pragma omp for schedule(runtime) reduction(+:totalcost)
    for (data_coord_t i = 0; i < mat->nrows; i++) {
      const opt_prec_t *myZ1 = Z1 + (i * LDIM);
      opt_prec_t *myZ1U = Z1update + (i * LDIM);
      
      memset(myZ1U, 0, sizeof(opt_prec_t) * LDIM);
      
      data_point_t start = mat->crs_ptrs[i];
      data_point_t end = mat->crs_ptrs[i + 1];

      data_point_t ptr;      
      for (ptr = start; ptr < end - 3; ptr += 4) {
	myZ2_1  = Z2 + (mat->crs_colids[ptr] * LDIM);
	myZ2_2  = Z2 + (mat->crs_colids[ptr + 1] * LDIM);
	myZ2_3  = Z2 + (mat->crs_colids[ptr + 2] * LDIM);
	myZ2_4  = Z2 + (mat->crs_colids[ptr + 3] * LDIM);

	diffs = _mm256_set_pd(-(mat->crs_values[ptr+3]), -(mat->crs_values[ptr+2]), -(mat->crs_values[ptr+1]), -(mat->crs_values[ptr]));
	for (int k = 0; k < LDIM; k++) {
	  diffs = _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(myZ1[k]), _mm256_set_pd(myZ2_4[k], myZ2_3[k], myZ2_2[k], myZ2_1[k])), diffs);
	}
	_mm256_storeu_pd(div_term + ptr, diffs);

	for (int k = 0; k < LDIM; k++) {
	  z2s = _mm256_mul_pd(diffs, _mm256_set_pd(myZ2_4[k],  myZ2_3[k], myZ2_2[k],  myZ2_1[k]));
	  z2s = _mm256_hadd_pd(z2s, z2s);
	  myZ1U[k] += ((*z2s0) + (*z2s2)) * scfac;
	}

	diffs = _mm256_mul_pd(diffs, diffs);
	diffs = _mm256_hadd_pd(diffs, diffs);
	totalcost += (*diffs0) + (*diffs2);
      }
      
      for (; ptr < end; ptr++) {
	opt_prec_t *myZ2_1  = Z2 + (mat->crs_colids[ptr] * LDIM);
	
	diff_1 = -(mat->crs_values[ptr]);
	for (int k = 0; k < LDIM; k++) {
	  diff_1 += myZ1[k] * myZ2_1[k];
	}
	
	div_term[ptr] = diff_1;
	opt_prec_t coef = diff_1 * scfac;
	for (int k = 0; k < LDIM; k++) {
	  myZ1U[k] += myZ2_1[k] * coef;
	}
	totalcost += diff_1 * diff_1;
      }
    }
  }
  return totalcost * 0.5 * scfac;
}

void Z2update(SparseMatrix* mat, opt_prec_t* Z1, opt_prec_t* Z2update) {
  opt_prec_t scfac = ((double)(1.0f)) / mat->nnnz;

#pragma omp parallel for schedule(runtime)
  for (data_coord_t j = 0; j < mat->ncols; j++) {
    opt_prec_t *  myZ2U = Z2update + (j * LDIM);
    memset (myZ2U, 0, sizeof(opt_prec_t) * LDIM);
    
    data_point_t start = mat->ccs_ptrs[j];
    data_point_t end = mat->ccs_ptrs[j + 1];
    data_point_t ptr;
    for (ptr = start; ptr < end - 3; ptr += 4) {
      const opt_prec_t *myZ1_1  = Z1 + (mat->ccs_rowids[ptr] * LDIM);
      const opt_prec_t *myZ1_2  = Z1 + (mat->ccs_rowids[ptr + 1] * LDIM);
      const opt_prec_t *myZ1_3  = Z1 + (mat->ccs_rowids[ptr + 2] * LDIM);
      const opt_prec_t *myZ1_4  = Z1 + (mat->ccs_rowids[ptr + 3] * LDIM);

      const opt_prec_t cv_1 = div_term[mat->ccs_translator[ptr]];
      const opt_prec_t cv_2 = div_term[mat->ccs_translator[ptr + 1]];
      const opt_prec_t cv_3 = div_term[mat->ccs_translator[ptr + 2]];
      const opt_prec_t cv_4 = div_term[mat->ccs_translator[ptr + 3]];

      for (int k = 0; k < LDIM; k++) {
	myZ2U[k] += (myZ1_1[k] * cv_1 + myZ1_2[k] * cv_2 + myZ1_3[k] * cv_3 + myZ1_4[k] * cv_4) * scfac;
      }
    }

    for (; ptr < end; ptr++) {
      opt_prec_t *myZ1 = Z1 + (mat->ccs_rowids[ptr] * LDIM);
      
      opt_prec_t cv = div_term[mat->ccs_translator[ptr]] * scfac;
      for (int k = 0; k < LDIM; k++) {
	myZ2U[k] += myZ1[k] * cv;
      }
    }
  }
}
#endif

//x is the solution, g is the gradient
void pmb_function(SparseMatrix* mat, opt_prec_t* x, opt_prec_t& f, opt_prec_t* g) {

  opt_prec_t * Z1 = &(x[0]);
  opt_prec_t * Z2 = &(x[mat->nrows * LDIM]);

#ifdef DOAEBUG
  cout << "\tFunction computation started... ";
#endif

  //divide g into G1 and G2
  opt_prec_t * G1 = &(g[0]);
  opt_prec_t * G2 = &(g[mat->nrows * LDIM]);
  
  //Compute the gradient and the objective function
#ifdef TIMER
  double time = omp_get_wtime();
#endif

  f = dTerm_Z1update(mat, Z1, G1, Z2);

#ifdef TIMER
  divTerm_Z1Time += omp_get_wtime() - time;
  time = omp_get_wtime();
#endif

  Z2update(mat, Z1, G2);

#ifdef TIMER
  Z2Time += omp_get_wtime() - time;
#endif
#ifdef DEBUG
  cout << " Done." << endl;
#endif
}

#endif
