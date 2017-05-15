#include <iostream>
#include <random>
#include "pmb_driver.hpp"
#include "sparse_mf_func.hpp"

using namespace std;

extern opt_prec_t scfac;
extern opt_prec_t* div_term;
extern double divTerm_Z1Time;
extern double Z2Time;
extern int LDIM;

int loadDataSparse(char* fileName, SparseMatrix*& mat) {
  FILE *mat_file;
#ifdef DEBUG
  cout << "Reading file " << fileName << endl;
#endif

  if((mat_file = fopen(fileName, "r")) == NULL) {
    cout << "Cannot read file\n";
    return 0;
  }

  int nrow, ncol, nnnz;
  fscanf(mat_file, "%d %d %d", &nrow, &ncol, &nnnz);

  data_coord_t* I = new data_coord_t[nnnz];
  data_coord_t* J = new data_coord_t[nnnz];
  data_val_t* V = new data_val_t[nnnz];

  for (int i = 0; i < nnnz; i++) {
    double temp_i, temp_j, temp_val;
    fscanf(mat_file, "%lf %lf %lf", &temp_i, &temp_j, &temp_val);
    I[i] = (data_coord_t)(temp_i - 1);
    J[i] = (data_coord_t)(temp_j - 1);
    V[i] = (data_val_t)(temp_val);
  }
  fclose(mat_file);

#ifdef DEBUG
  cout << "File is read. Allocating memory for matrix " << endl;
#endif

  mat = new SparseMatrix(nrow, ncol, nnnz);

#ifdef DEBUG
  cout << "Memory is allocated; creating crs and ccs" << endl;
#endif

  memset(mat->crs_ptrs, 0, (nrow + 1) * sizeof(data_point_t));

  // in each cell of the array crs_ptrs, we have number of elements in that row in the matrix, but crs_ptrs is one cell ahead.
  for(data_point_t i = 0; i < nnnz; i++) {
    mat->crs_ptrs[I[i]+1]++; //increase the counts
  }

  //Now we have cumulative ordering of crs_ptrs.
  for(data_coord_t i = 1; i <= nrow; i++) {
    mat->crs_ptrs[i] += mat->crs_ptrs[i-1]; //prefix sum
  }

  //here we set crs_colids such that for each element, it holds the related column of that element
  for(data_point_t i = 0; i < nnnz; i++) {
    data_coord_t rowid = I[i];
    data_point_t index = mat->crs_ptrs[rowid];

    mat->crs_colids[index] = J[i];
    mat->crs_values[index] = V[i];

    mat->crs_ptrs[rowid] = mat->crs_ptrs[rowid] + 1;
  }

  //forward shift and assign for fixing the ptrs array
  for(data_coord_t i = nrow; i > 0; i--) {
    mat->crs_ptrs[i] = mat->crs_ptrs[i-1];
  }
  mat->crs_ptrs[0] = 0;

#ifdef DEBUG
  cout << "\tcrs is created" << endl;
#endif

  memset(mat->ccs_ptrs, 0, (ncol + 1) * sizeof(data_point_t));

  for(data_point_t i = 0; i < nnnz; i++) {
    mat->ccs_ptrs[J[i]+1]++;
  }

  for(data_coord_t i = 1; i <= ncol; i++) {
    mat->ccs_ptrs[i] += mat->ccs_ptrs[i-1];
  }

  for(data_coord_t i = 0; i < nrow; i++) {
    for(data_point_t ptr = mat->crs_ptrs[i]; ptr < mat->crs_ptrs[i+1]; ptr++) {
      data_coord_t colid  = mat->crs_colids[ptr];

      data_point_t index = mat->ccs_ptrs[colid];
      mat->ccs_rowids[index] = i;
      mat->ccs_translator[index] = ptr;

      mat->ccs_ptrs[colid] = mat->ccs_ptrs[colid] + 1;
    }
  }

  for(data_coord_t i = ncol; i > 0; i--) {
    mat->ccs_ptrs[i] = mat->ccs_ptrs[i-1];
  }
  mat->ccs_ptrs[0] = 0;

#ifdef DEBUG
  cout << "\tccs is created" << endl;
#endif

  delete[] I;
  delete[] J;
  delete[] V;

  return 1;
}

int main(int argc, char * argv[]) {
  if(argc != 3) {
    cout << "Usage: executable filename latent_dimension" << endl;
    return 0;
  }

  //load data
  char fileName[80];
  strcpy (fileName, argv[1]);
  LDIM = atoi(argv[2]);

  SparseMatrix* mat;
  if(!loadDataSparse(fileName, mat)) {
    return 0;
  }
  scfac = (1.0 / mat->nnnz);

  //this is the place we store errors; global scope
  div_term = new opt_prec_t[mat->nnnz];

  cout << "rows/cols/ratings: " << mat->nrows << " " << mat->ncols << " " << mat->nnnz << endl;

  //initial solution
  std::random_device r;
  std::default_random_engine eng(r());
  std::uniform_int_distribution<int> uniform_dist(1, 5);
  std::uniform_real_distribution<> unif(1,5);
  opt_prec_t* x_0 = new opt_prec_t[mat->n];
  for (int i = 0; i < mat->n; i++) {
    x_0[i] = sqrt(unif(eng)/LDIM);
  }

  //options
  Options options;
  options.tol = 1e-05;
  options.maxiter = 500;
  options.maxinneriter = 100;
  options.M = 5;

  Output output;
  double tt = omp_get_wtime();
  pmb_driver<SparseMatrix>(mat, x_0, options, output, mat->n);
  double timeSpent = omp_get_wtime() - tt;

#ifdef TIMER
  cout << "  dt_Z1   time:\t" << divTerm_Z1Time << endl;
  cout << "     Z2   time:\t" << Z2Time << endl;
  cout << " -----------------------------" << endl << endl;
#endif

  cout << "Exit: " << output.exit << endl;
  cout << "Fval: " << output.fval << endl;
  cout << "Norm: " << output.gradient_norm << endl;
  cout << "Iterations: " << output.iterations << endl;
  cout << "Evaluations: " << output.evaluations << endl;
  cout << "Time Spent in Miliseconds: " << timeSpent << endl;
  cout << "RMSE: " << pow(((output.fval * 2) ), 0.5) << endl;
  cout << "Latent Dim: " << LDIM << endl;

  delete mat;
  delete[] div_term;

  return 0;
}
