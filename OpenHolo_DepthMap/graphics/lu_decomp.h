#ifndef __lu_decomp_h
#define __lu_decomp_h

namespace graphics {

int Crout_LU_with_Pivoting_Solve(double *LU, double B[], int pivot[], double x[], int n);
int Crout_LU_Decomposition_with_Pivoting(double *A, int pivot[], int n);

int Crout_LU_Decomposition(double *A, int n);
int Crout_LU_Solve(double *LU, double B[], double x[], int n);
void alloc_matrix(double*& A, double**& pA, int m, int n);
int Gaussian_Elimination(double *A, int n, double *B);

int Choleski_LU_Decomposition(double *A, int n);
int Singular_Value_Decomposition(double* A, int nrows, int ncols, double* U, 
                      double* singular_values, double* V, double* dummy_array);
};

#endif