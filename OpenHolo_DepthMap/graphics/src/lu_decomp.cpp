#include <stdlib.h>

namespace graphics {
////////////////////////////////////////////////////////////////////////////////
// File: lower_triangular.c                                                   //
// Routines:                                                                  //
//    Lower_Triangular_Solve                                                  //
//    Lower_Triangular_Inverse                                                //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  int Lower_Triangular_Solve(double *L, double *B, double x[], int n)       //
//                                                                            //
//  Description:                                                              //
//     This routine solves the linear equation Lx = B, where L is an n x n    //
//     lower triangular matrix.  (The superdiagonal part of the matrix is     //
//     not addressed.)                                                        //
//     The algorithm follows:                                                 //
//                      x[0] = B[0]/L[0][0], and                              //
//     x[i] = [B[i] - (L[i][0] * x[0]  + ... + L[i][i-1] * x[i-1])] / L[i][i],//
//     for i = 1, ..., n-1.                                                   //
//                                                                            //
//  Arguments:                                                                //
//     double *L   Pointer to the first element of the lower triangular       //
//                 matrix.                                                    //
//     double *B   Pointer to the column vector, (n x 1) matrix, B.           //
//     double *x   Pointer to the column vector, (n x 1) matrix, x.           //
//     int     n   The number of rows or columns of the matrix L.             //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix L is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N], B[N], x[N];                                            //
//                                                                            //
//     (your code to create matrix A and column vector B)                     //
//     err = Lower_Triangular_Solve(&A[0][0], B, x, n);                       //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else printf(" The solution is \n");                                    //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Lower_Triangular_Solve(double *L, double B[], double x[], int n)
{
   int i, k;

//         Solve the linear equation Lx = B for x, where L is a lower
//         triangular matrix.                                      
   
   for (k = 0; k < n; L += n,++k) {
      if (*(L + k) == 0.0) return -1;           // The matrix L is singular
      x[k] = B[k];
      for (i = 0; i < k;++i)x[k] -= x[i] * *(L + i);
      x[k] /= *(L + k);
   }

   return 0;
}


////////////////////////////////////////////////////////////////////////////////
//  int Lower_Triangular_Inverse(double *L,  int n)                           //
//                                                                            //
//  Description:                                                              //
//     This routine calculates the inverse of the lower triangular matrix L.  //
//     The superdiagonal part of the matrix is not addressed.                 //
//     The algorithm follows:                                                 //
//        Let M be the inverse of L, then L M = I,                            //
//     M[i][i] = 1.0 / L[i][i] for i = 0, ..., n-1, and                       //
//     M[i][j] = -[(L[i][j] M[j][j] + ... + L[i][i-1] M[i-1][j])] / L[i][i],  //
//     for i = 1, ..., n-1, j = 0, ..., i - 1.                                //
//                                                                            //
//                                                                            //
//  Arguments:                                                                //
//     double *L   On input, the pointer to the first element of the matrix   //
//                 whose lower triangular elements form the matrix which is   //
//                 to be inverted. On output, the lower triangular part is    //
//                 replaced by the inverse.  The superdiagonal elements are   //
//                 not modified.                                              //
//                 its inverse.                                               //
//     int     n   The number of rows and/or columns of the matrix L.         //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix L is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double L[N][N];                                                        //
//                                                                            //
//     (your code to create the matrix L)                                     //
//     err = Lower_Triangular_Inverse(&L[0][0], N);                           //
//     if (err < 0) printf(" Matrix L is singular\n");                        //
//     else {                                                                 //
//        printf(" The inverse is \n");                                       //
//           ...                                                              //
//     }                                                                      //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Lower_Triangular_Inverse(double *L, int n)
{
   int i, j, k;
   double *p_i, *p_j, *p_k;
   double sum;

//         Invert the diagonal elements of the lower triangular matrix L.

   for (k = 0, p_k = L; k < n; p_k += (n + 1),++k) {
      if (*p_k == 0.0) return -1;
      else *p_k = 1.0 / *p_k;
   }

//         Invert the remaining lower triangular matrix L row by row.

   for (i = 1, p_i = L + n; i < n;++i, p_i += n) {
      for (j = 0, p_j = L; j < i; p_j += n, ++j) {
         sum = 0.0;
         for (k = j, p_k = p_j; k < i;++k, p_k += n)
            sum += *(p_i + k) * *(p_k + j);
         *(p_i + j) = - *(p_i + i) * sum;
      }
   }
  
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
// File: unit_upper_triangular.c                                              //
// Routines:                                                                  //
//    Unit_Upper_Triangular_Solve                                             //
//    Unit_Upper_Triangular_Inverse                                           //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  int Unit_Upper_Triangular_Solve(double *U, double *B, double x[], int n)  //
//                                                                            //
//  Description:                                                              //
//     This routine solves the linear equation Ux = B, where U is an n x n    //
//     unit upper triangular matrix.  (Only the superdiagonal part of the     //
//     matrix is addressed.)  The diagonal is assumed to consist of 1's and   //
//     is not addressed.                                                      //
//     The algorithm follows:                                                 //
//                  x[n-1] = B[n-1], and                                      //
//       x[i] = B[i] - (U[i][i+1] * x[i+1]  + ... + U[i][n-1] * x[n-1]),      //
//     for i = n-2, ..., 0.                                                   //
//                                                                            //
//  Arguments:                                                                //
//     double *U   Pointer to the first element of the upper triangular       //
//                 matrix.                                                    //
//     double *B   Pointer to the column vector, (n x 1) matrix, B.           //
//     double *x   Pointer to the column vector, (n x 1) matrix, x.           //
//     int     n   The number of rows or columns of the matrix U.             //
//                                                                            //
//  Return Values:                                                            //
//     void                                                                   //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N], B[N], x[N];                                            //
//                                                                            //
//     (your code to create matrix A and column vector B)                     //
//     Unit_Upper_Triangular_Solve(&A[0][0], B, x, n);                        //
//     printf(" The solution is \n");                                         //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
void Unit_Upper_Triangular_Solve(double *U, double B[], double x[], int n)
{
   int i, k;

//         Solve the linear equation Ux = B for x, where U is an upper
//         triangular matrix.                                      
   x[n-1] = B[n-1]; 
   for (k = n-2, U += n * (n - 2); k >= 0; U -= n, k--) {
      x[k] = B[k];
      for (i = k + 1; i < n;++i)x[k] -= x[i] * *(U + i);
   }
}


////////////////////////////////////////////////////////////////////////////////
//  int Unit_Upper_Triangular_Inverse(double *U,  int n)                      //
//                                                                            //
//  Description:                                                              //
//     This routine calculates the inverse of the unit upper triangular matrix//
//     U.  The subdiagonal part of the matrix is not addressed.               //
//     The diagonal is assumed to consist of 1's and is not addressed.        //
//     The algorithm follows:                                                 //
//        Let M be the inverse of U, then U M = I,                            //
//          M[i][j] = -( U[i][i+1] M[i+1][j] + ... + U[i][j] M[j][j] ),       //
//     for i = n-2, ... , 0,  j = n-1, ..., i+1.                              //
//                                                                            //
//                                                                            //
//  Arguments:                                                                //
//     double *U   On input, the pointer to the first element of the matrix   //
//                 whose unit upper triangular elements form the matrix which //
//                 is to be inverted. On output, the upper triangular part is //
//                 replaced by the inverse.  The subdiagonal elements are     //
//                 not modified.                                              //
//     int     n   The number of rows and/or columns of the matrix U.         //
//                                                                            //
//  Return Values:                                                            //
//     void                                                                   //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double U[N][N];                                                        //
//                                                                            //
//     (your code to create the matrix U)                                     //
//     Unit_Upper_Triangular_Inverse(&U[0][0], N);                            //
//     printf(" The inverse is \n");                                          //
//           ...                                                              //
//     }                                                                      //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
void Unit_Upper_Triangular_Inverse(double *U, int n)
{
   int i, j, k;
   double *p_i, *p_j, *p_k;
   double sum;

//         Invert the superdiagonal part of the matrix U row by row where
//         the diagonal elements are assumed to be 1.0.

   for (i = n - 2, p_i = U + n * (n - 2); i >=0; p_i -= n, i-- ) {
      for (j = n - 1; j > i; j--) {
         *(p_i + j) = -*(p_i + j);
         for (k = i + 1, p_k = p_i + n; k < j; p_k += n,++k ) 
            *(p_i + j) -= *(p_i + k) * *(p_k + j);
      }
   }
}


////////////////////////////////////////////////////////////////////////////////
// File: tridiagonal.c                                                        //
// Contents:                                                                  //
//    Tridiagonal_LU_Decomposition                                            //
//    Tridiagonal_LU_Solve                                                    //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Tridiagonal_LU_Decomposition(double *subdiagonal, double *diagonal,       //
//                                           double *superdiagonal, int n )   //
//                                                                            //
//  Description:                                                              //
//     This routine decomposes a tridiagonal matrix A into the product of     //
//     a unit lower triangular (bidiagonal) matrix L and an upper triangular  //
//     (bidiagonal) matrix U, A = LU.                                         //
//     The tridiagonal matrix A is defined by the three vectors, subdiagonal, //
//     diagonal, and superdiagonal, where the i-th component of subdiagonal is//
//     subdiagonal[i] = A[i+1][i], for i = 0, ..., n - 2; the i-th component  //
//     of diagonal is diagonal[i] = A[i][i], for i = 0, ..., n - 1; and the   //
//     i-th component of superdiagonal is superdiagonal[i] = A[i][i+1], for   //
//     i = 0, ..., n - 2.                                                     //
//     The algorithm proceeds by decomposing the matrix A into the product    //
//     of a unit lower triangular (bidiagonal) matrix, stored in subdiagonal, //
//     and an upper triangular (bidiagonal) matrix, stored in diagonal and    //
//     and superdiagonal.                                                     //
//     After performing the LU decomposition for A, call Tridiagonal_LU_Solve //
//     to solve the equation Ax = B for x given B.                            //
//                                                                            //
//     This routine can fail if A[0][0] = 0 or if during the LU decomposition //
//     the diagonal element of U becomes 0.  This does not imply that the     //
//     matrix A is singular.  If A is positive definite or if A is diagonally //
//     dominant then the procedure should not fail.                           //
//                                                                            //
//  Arguments:                                                                //
//     double subdiagonal[]                                                   //
//        On input, subdiagonal[i] is the subdiagonal element A[i+1][i].      //
//        On output, subdiagonal[i] is the subdiagonal of the unit lower      //
//        triangular matrix L in the LU decomposition of A.                   //
//     double diagonal[]                                                      //
//        On input, diagonal[i] is the diagonal element A[i][i] of the matrix //
//        A.  On output, diagonal[i] is the diagonal of the upper triangular  //
//        matrix U in the LU decomposition of A.                              //
//     double superdiagonal[]                                                 //
//        On input, superdiagonal[i] is the superdiagonal element A[i][i+1] of//
//        the matrix A.  On output, superdiagonal[i] is the superdiagonal of  //
//        the upper triangular matrix U, which agrees with the input.         //
//     int     n   The number of rows and/or columns of the matrix A.         //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - A zero occurred on the diagonal of U.                     //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double subdiagonal[N], diagonal[N], superdiagonal[N];                  //
//                                                                            //
//     (your code to create subdiagonal, diagonal, and superdiagonal)         //
//     err = Tridiagonal_LU_Decomposition(subdiagonal, diagonal,              //
//                                                         superdiagonal, N); //
//     if (err < 0) printf(" Matrix A failed the LU decomposition\n");        //
//     else { printf(" The Solution is: \n"); ...                             //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Tridiagonal_LU_Decomposition( double subdiagonal[], double diagonal[],
                                                double superdiagonal[], int n )
{
   int i;

   for (i = 0; i < (n-1);++i){
      if (diagonal[i] == 0.0) return -1;
      subdiagonal[i] /= diagonal[i];
      diagonal[i+1] -= subdiagonal[i] * superdiagonal[i];
   }
   if (diagonal[n-1] == 0.0) return -1;      
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
// int Tridiagonal_LU_Solve(double subdiagonal[], double diagonal[],          //
//                 double superdiagonal[],  double B[], double x[],  int n)   //
//                                                                            //
//  Description:                                                              //
//     This routine uses the LU decomposition from the routine above,         //
//     Tridiagonal_LU_Decomposition, to solve the linear equation Ax = B,     //
//     where A = LU, L is the unit lower triangular (bidiagonal) matrix with  //
//     subdiagonal subdiagonal[] and diagonal all 1's, and U is the upper     //
//     triangular (bidiagonal) matrix with diagonal diagonal[] and            //
//     superdiagonal superdiagonal[].                                         //
//     The solution proceeds by solving the linear equation Ly = B for y and  //
//     subsequently solving the linear equation Ux = y for x.                 //
//                                                                            //
//  Arguments:                                                                //
//     double subdiagonal[]                                                   //
//        The subdiagonal of the unit lower triangular matrix L in the LU     //
//        decomposition of A.                                                 //
//     double diagonal[]                                                      //
//        The diagonal of the upper triangular matrix U in the LU decomposi-  //
//        tion of A.                                                          //
//     double superdiagonal[]                                                 //
//        The superdiagonal of the upper triangular matrix U.                 //
//     double B[]                                                             //
//        Pointer to the column vector, (n x 1) matrix, B.                    //
//     double x[]                                                             //
//        Solution to the equation Ax = B.                                    //
//     int     n   The number of rows and/or columns of the matrix LU.        //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix U is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double subdiagonal[N], diagonal[N], superdiagonal[N];                  //
//     double B[N], x[N];                                                     //
//                                                                            //
//     (your code to create subdiagonal, diagonal, superdiagonal, and B)      //
//     err = Tridiagonal_LU_Decomposition(subdiagonal, diagonal,              //
//                                                          superdiagonal, N);//
//     if (err < 0) printf(" Matrix A is fails the LU decomposition\n");      //
//     else {                                                                 //
//        err = Tridiagonal_LU_Solve(subdiagona, diagonal, superdiagonal, B,  //
//                                                                      x, n);//
//        if (err < 0) printf(" Matrix A is singular\n");                     //
//        else printf(" The solution is \n");                                 //
//           ...                                                              //
//     }                                                                      //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Tridiagonal_LU_Solve( double subdiagonal[], double diagonal[],
                       double superdiagonal[], double B[], double x[], int n)
{
   int i;

//         Check that all diagonal elements are nonzero.
//         If a diagonal element is zero then U is singular, so return
//         signalling an error.

   for (i = 0; i < n;++i)if (diagonal[i] == 0.0) return -1;

//         Solve the linear equation Ly = B for y, where L is a lower
//         triangular matrix.
   
   x[0] = B[0];
   for (i = 1; i < n;++i)x[i] = B[i] - subdiagonal[i-1] * x[i-1];

//         Solve the linear equation Ux = y, where y is the solution
//         obtained above of Ly = B and U is an upper triangular matrix.

   x[n-1] /= diagonal[n-1];

   for (i = n-2; i >= 0; i--) {
      x[i] -= superdiagonal[i] * x[i+1];
      x[i] /= diagonal[i];
   }
   
   return 0;
} 



////////////////////////////////////////////////////////////////////////////////
// File: crout.c                                                              //
// Routines:                                                                  //
//    Crout_LU_Decomposition                                                  //
//    Crout_LU_Solve                                                          //
//                                                                            //
// Required Externally Defined Routines:                                      //
//    Lower_Triangular_Solve                                                  //
//    Unit_Upper_Triangular_Solve                                             //
////////////////////////////////////////////////////////////////////////////////

//                    Required Externally Defined Routines 
int  Lower_Triangular_Solve(double *L, double B[], double x[], int n);
void Unit_Upper_Triangular_Solve(double *U, double B[], double x[], int n);

////////////////////////////////////////////////////////////////////////////////
//  int Crout_LU_Decomposition(double *A, int n)                              //
//                                                                            //
//  Description:                                                              //
//     This routine uses Crout's method to decompose the n x n matrix A       //
//     into a lower triangular matrix L and a unit upper triangular matrix U  //
//     such that A = LU.                                                      //
//     The matrices L and U replace the matrix A so that the original matrix  //
//     A is destroyed.                                                        //
//     Note!  In Crout's method the diagonal elements of U are 1 and are      //
//            not stored.                                                     //
//     Note!  The determinant of A is the product of the diagonal elements    //
//            of L.  (det A = det L * det U = det L).                         //
//     This routine is suitable for those classes of matrices which when      //
//     performing Gaussian elimination do not need to undergo partial         //
//     pivoting, e.g. positive definite symmetric matrices, diagonally        //
//     dominant band matrices, etc.                                           //
//     For the more general case in which partial pivoting is needed use      //
//                    Crout_LU_Decomposition_with_Pivoting.                   //
//     The LU decomposition is convenient when one needs to solve the linear  //
//     equation Ax = B for the vector x while the matrix A is fixed and the   //
//     vector B is varied.  The routine for solving the linear system Ax = B  //
//     after performing the LU decomposition for A is Crout_LU_Solve          //
//     (see below).                                                           //
//                                                                            //
//     The Crout method is given by evaluating, in order, the following       //
//     pair of expressions for k = 0, ... , n - 1:                            //
//       L[i][k] = (A[i][k] - (L[i][0]*U[0][k] + . + L[i][k-1]*U[k-1][k]))    //
//                                 for i = k, ... , n-1,                      //
//       U[k][j] = A[k][j] - (L[k][0]*U[0][j] + ... + L[k][k-1]*U[k-1][j])    //
//                                                                  / L[k][k] //
//                                      for j = k+1, ... , n-1.               //
//       The matrix U forms the upper triangular matrix, and the matrix L     //
//       forms the lower triangular matrix.                                   //
//                                                                            //
//  Arguments:                                                                //
//     double *A   Pointer to the first element of the matrix A[n][n].        //
//     int     n   The number of rows or columns of the matrix A.             //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix A is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N];                                                        //
//                                                                            //
//     (your code to intialize the matrix A)                                  //
//                                                                            //
//     err = Crout_LU_Decomposition(&A[0][0], N);                             //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else { printf(" The LU decomposition of A is \n");                     //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Crout_LU_Decomposition(double *A, int n)
{
   int row, i, j, k, p;
   double *p_k, *p_row, *p_col;

//         For each row and column, k = 0, ..., n-1,
//            find the lower triangular matrix elements for column k
//            and if the matrix is non-singular (nonzero diagonal element).
//            find the upper triangular matrix elements for row k. 
 
   for (k = 0, p_k = A; k < n; p_k += n,++k) {
      for (i = k, p_row = p_k; i < n; p_row += n,++i){
         for (p = 0, p_col = A; p < k; p_col += n, p++)
            *(p_row + k) -= *(p_row + p) * *(p_col + k);
      }  
      //if ( *(p_k + k) == 0.0 ) return -1;
      for (j = k+1; j < n; ++j) {
         for (p = 0, p_col = A; p < k; p_col += n,  p++)
            *(p_k + j) -= *(p_k + p) * *(p_col + j);
         *(p_k + j) /= *(p_k + k);
      }
   }
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
//  int Crout_LU_Solve(double *LU, double *B, double *x,  int n)              //
//                                                                            //
//  Description:                                                              //
//     This routine uses Crout's method to solve the linear equation Ax = B.  //
//     This routine is called after the matrix A has been decomposed into a   //
//     product of a lower triangular matrix L and a unit upper triangular     //
//     matrix U without pivoting.  The argument LU is a pointer to the matrix //
//     the superdiagonal part of which is U and the subdiagonal together with //
//     the diagonal part is L. (The diagonal part of U is 1 and is not        //
//     stored.)   The matrix A = LU.                                          //
//     The solution proceeds by solving the linear equation Ly = B for y and  //
//     subsequently solving the linear equation Ux = y for x.                 //
//                                                                            //
//  Arguments:                                                                //
//     double *LU  Pointer to the first element of the matrix whose elements  //
//                 form the lower and upper triangular matrix factors of A.   //
//     double *B   Pointer to the column vector, (n x 1) matrix, B            //
//     double *x   Solution to the equation Ax = B.                           //
//     int     n   The number of rows or columns of the matrix LU.            //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix A is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N], B[N], x[N];                                            //
//                                                                            //
//     (your code to create matrix A and column vector B)                     //
//     err = Crout_LU_Decomposition(&A[0][0], N);                             //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else {                                                                 //
//        err = Crout_LU_Solve(&A[0][0], B, x, n);                            //
//        if (err < 0) printf(" Matrix A is singular\n");                     //
//        else printf(" The solution is \n");                                 //
//           ...                                                              //
//     }                                                                      //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Crout_LU_Solve(double *LU, double B[], double x[], int n)
{

//         Solve the linear equation Lx = B for x, where L is a lower
//         triangular matrix.                                      
   
   if ( Lower_Triangular_Solve(LU, B, x, n) < 0 ) return -1;

//         Solve the linear equation Ux = y, where y is the solution
//         obtained above of Lx = B and U is an upper triangular matrix.
//         The diagonal part of the upper triangular part of the matrix is
//         assumed to be 1.0.

   Unit_Upper_Triangular_Solve(LU, x, x, n);
  
   return 0;
}

void alloc_matrix(double*& A, double**& pA, int m, int n)
{
	//double **pA;
	A = (double*) malloc(sizeof(double) * m * n);
	pA = (double**) malloc(sizeof(double) * m);
	int i = 1;
	for(*pA = A; i < m;++i)
		(double*)(*(pA+i)) = 
		(double*)(*(pA + i - 1) + n);
}


////////////////////////////////////////////////////////////////////////////////
//  int Gaussian_Elimination(double *A, int n, double *B)                     //
//                                                                            //
//     Solve the linear system of equations AX=B where A is an n x n matrix   //
//     B is an n-dimensional column vector (n x 1 matrix) for the             //
//     n-dimensional column vector (n x 1 matrix) X.                          //
//                                                                            //
//     This routine performs partial pivoting and the elements of A are       //
//     modified during computation.  The result X is returned in B.           //
//     If the matrix A is singular, the return value of the function call is  //
//     -1. If the solution was found, the function return value is 0.         //
//                                                                            //
//  Arguments:                                                                //
//     double *A      On input, the pointer to the first element of the       //
//                    matrix A[n][n].  On output, the matrix A is destroyed.  //
//     int     n      The number of rows and columns of the matrix A and the  //
//                    dimension of B.                                         //
//     double *B      On input, the pointer to the first element of the       //
//                    vector B[n].  On output, the vector B is replaced by the//
//                    vector X, the solution of AX = B.                       //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix A is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N], B[N];                                                  //
//                                                                            //
//     (your code to create the matrix A and vector B )                       //
//     err = Gaussian_Elimination((double*)A, NROWS, B);                      //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else { printf(" The Solution is: \n"); ...                             //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
#include <math.h>                                     // required for fabs()

int Gaussian_Elimination(double *A, int n, double *B)
{
   int row, i, j, pivot_row;
   double max, dum, *pa, *pA, *A_pivot_row;

      // for each variable find pivot row and perform forward substitution

   pa = A;
   for (row = 0; row < (n - 1); row++, pa += n) {

                       //  find the pivot row

      A_pivot_row = pa;
      max = fabs(*(pa + row));
      pA = pa + n;
      pivot_row = row;
      for (i = row + 1; i < n; pA += n,++i)
         if ((dum = fabs(*(pA + row))) > max) { 
            max = dum; A_pivot_row = pA; pivot_row = i; 
         }
      if (max == 0.0) return -1;                // the matrix A is singular

        // and if it differs from the current row, interchange the two rows.
             
      if (pivot_row != row) {
         for (i = row; i < n;++i){
            dum = *(pa + i);
            *(pa + i) = *(A_pivot_row + i);
            *(A_pivot_row + i) = dum;
         }
         dum = B[row];
         B[row] = B[pivot_row];
         B[pivot_row] = dum;
      }

                      // Perform forward substitution

      for (i = row + 1; i < n;++i){
         pA = A + i * n;
         dum = - *(pA + row) / *(pa + row);
         *(pA + row) = 0.0;
         for (j = row + 1; j < n; ++j) *(pA + j) += dum * *(pa + j);
         B[i] += dum * B[row];
      }
   }

                    // Perform backward substitution
  
   pa = A + (n - 1) * n;
   for (row = n - 1; row >= 0; pa -= n, row--) {
      if ( *(pa + row) == 0.0 ) return -1;           // matrix is singular
      dum = 1.0 / *(pa + row);
      for ( i = row + 1; i < n;++i)*(pa + i) *= dum; 
      B[row] *= dum; 
      for ( i = 0, pA = A; i < row; pA += n,++i){
         dum = *(pA + row);
         for ( j = row + 1; j < n; ++j) *(pA + j) -= dum * *(pa + j);
         B[i] -= dum * B[row];
      }
   }
   return 0;
}
////////////////////////////////////////////////////////////////////////////////
// File: crout_pivot.c                                                        //
// Routines:                                                                  //
//    Crout_LU_Decomposition_with_Pivoting                                    //
//    Crout_LU_with_Pivoting_Solve                                            //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  int Crout_LU_Decomposition_with_Pivoting(double *A, int pivot[], int n)   //
//                                                                            //
//  Description:                                                              //
//     This routine uses Crout's method to decompose a row interchanged       //
//     version of the n x n matrix A into a lower triangular matrix L and a   //
//     unit upper triangular matrix U such that A = LU.                       //
//     The matrices L and U replace the matrix A so that the original matrix  //
//     A is destroyed.                                                        //
//     Note!  In Crout's method the diagonal elements of U are 1 and are      //
//            not stored.                                                     //
//     Note!  The determinant of A is the product of the diagonal elements    //
//            of L.  (det A = det L * det U = det L).                         //
//     The LU decomposition is convenient when one needs to solve the linear  //
//     equation Ax = B for the vector x while the matrix A is fixed and the   //
//     vector B is varied.  The routine for solving the linear system Ax = B  //
//     after performing the LU decomposition for A is                         //
//                      Crout_LU_with_Pivoting_Solve.                         //
//     (see below).                                                           //
//                                                                            //
//     The Crout method with partial pivoting is: Determine the pivot row and //
//     interchange the current row with the pivot row, then assuming that     //
//     row k is the current row, k = 0, ..., n - 1 evaluate in order the      //
//     the following pair of expressions                                      //
//       L[i][k] = (A[i][k] - (L[i][0]*U[0][k] + . + L[i][k-1]*U[k-1][k]))    //
//                                 for i = k, ... , n-1,                      //
//       U[k][j] = A[k][j] - (L[k][0]*U[0][j] + ... + L[k][k-1]*U[k-1][j])    //
//                                                                  / L[k][k] //
//                                      for j = k+1, ... , n-1.               //
//       The matrix U forms the upper triangular matrix, and the matrix L     //
//       forms the lower triangular matrix.                                   //
//                                                                            //
//  Arguments:                                                                //
//     double *A       Pointer to the first element of the matrix A[n][n].    //
//     int    pivot[]  The i-th element is the pivot row interchanged with    //
//                     row i.                                                 //
//     int     n       The number of rows or columns of the matrix A.         //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix A is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N];                                                        //
//     int    pivot[N];                                                       //
//                                                                            //
//     (your code to intialize the matrix A)                                  //
//                                                                            //
//     err = Crout_LU_Decomposition_with_Pivoting(&A[0][0], pivot, N);        //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else { printf(" The LU decomposition of A is \n");                     //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //

#include <math.h>                                     // required for fabs()

int Crout_LU_Decomposition_with_Pivoting(double *A, int pivot[], int n)
{
   int row, i, j, k, p;
   double *p_k, *p_row, *p_col;
   double max;

//         For each row and column, k = 0, ..., n-1,

   for (k = 0, p_k = A; k < n; p_k += n,++k) {
 
//            find the pivot row

      pivot[k] = k;
      max = fabs( *(p_k + k) );
      for (j = k + 1, p_row = p_k + n; j < n; j++, p_row += n) {
         if ( max < fabs(*(p_row + k)) ) {
            max = fabs(*(p_row + k));
            pivot[k] = j;
            p_col = p_row;
         }
      }

//            and if it differs from the current row, interchange the two rows.
   
      if (pivot[k] != k)
         for (j = 0; j < n; ++j) {
            max = *(p_k + j);
            *(p_k + j) = *(p_col + j);
            *(p_col + j) = max;
         }

//            find the lower triangular matrix elements for column k

      for (i = k, p_row = p_k; i < n; p_row += n,++i){
         for (p = 0, p_col = A; p < k; p_col += n, p++)
            *(p_row + k) -= *(p_row + p) * *(p_col + k);
      }  

//            and if the matrix is non-singular

      if ( *(p_k + k) == 0.0 ) return -1;

//            find the upper triangular matrix elements for row k. 
 
      for (j = k+1; j < n; ++j) {
         for (p = 0, p_col = A; p < k; p_col += n,  p++)
            *(p_k + j) -= *(p_k + p) * *(p_col + j);
         *(p_k + j) /= *(p_k + k);
      }
   }
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
//  int Crout_LU_with_Pivoting_Solve(double *LU, double B[], int pivot[],     //
//                                                        double x[], int n)  //
//                                                                            //
//  Description:                                                              //
//     This routine uses Crout's method to solve the linear equation Ax = B.  //
//     This routine is called after the matrix A has been decomposed into a   //
//     product of a lower triangular matrix L and a unit upper triangular     //
//     matrix U without pivoting.  The argument LU is a pointer to the matrix //
//     the superdiagonal part of which is U and the subdiagonal together with //
//     the diagonal part is L. (The diagonal part of U is 1 and is not        //
//     stored.)   The matrix A = LU.                                          //
//     The solution proceeds by solving the linear equation Ly = B for y and  //
//     subsequently solving the linear equation Ux = y for x.                 //
//                                                                            //
//  Arguments:                                                                //
//     double *LU      Pointer to the first element of the matrix whose       //
//                     elements form the lower and upper triangular matrix    //
//                     factors of A.                                          //
//     double *B       Pointer to the column vector, (n x 1) matrix, B.       //
//     int    pivot[]  The i-th element is the pivot row interchanged with    //
//                     row i.                                                 //
//     double *x       Solution to the equation Ax = B.                       //
//     int     n       The number of rows or columns of the matrix LU.        //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix A is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N], B[N], x[N];                                            //
//     int    pivot[N];                                                       //
//                                                                            //
//     (your code to create matrix A and column vector B)                     //
//     err = Crout_LU_Decomposition_with_Pivoting(&A[0][0], pivot, N);        //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else {                                                                 //
//        err = Crout_LU_with_Pivoting_Solve(&A[0][0], B, pivot, x, n);       //
//        if (err < 0) printf(" Matrix A is singular\n");                     //
//        else printf(" The solution is \n");                                 //
//           ...                                                              //
//     }                                                                      //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Crout_LU_with_Pivoting_Solve(double *LU, double B[], int pivot[], 
                                                            double x[], int n)
{
   int i, k;
   double *p_k;
   double dum;

//         Solve the linear equation Lx = B for x, where L is a lower
//         triangular matrix.                                      
   
   for (k = 0, p_k = LU; k < n; p_k += n,++k) {
      if (pivot[k] != k) {dum = B[k]; B[k] = B[pivot[k]]; B[pivot[k]] = dum; }
      x[k] = B[k];
      for (i = 0; i < k;++i)x[k] -= x[i] * *(p_k + i);
      x[k] /= *(p_k + k);
   }

//         Solve the linear equation Ux = y, where y is the solution
//         obtained above of Lx = B and U is an upper triangular matrix.
//         The diagonal part of the upper triangular part of the matrix is
//         assumed to be 1.0.

   for (k = n-1, p_k = LU + n*(n-1); k >= 0; k--, p_k -= n) {
      if (pivot[k] != k) {dum = B[k]; B[k] = B[pivot[k]]; B[pivot[k]] = dum; }
      for (i = k + 1; i < n;++i)x[k] -= x[i] * *(p_k + i);
      if (*(p_k + k) == 0.0) return -1;
   }
  
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
// File: upper_triangular.c                                                   //
// Routines:                                                                  //
//    Upper_Triangular_Solve                                                  //
//    Upper_Triangular_Inverse                                                //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  int Upper_Triangular_Solve(double *U, double *B, double x[], int n)       //
//                                                                            //
//  Description:                                                              //
//     This routine solves the linear equation Ux = B, where U is an n x n    //
//     upper triangular matrix.  (The subdiagonal part of the matrix is       //
//     not addressed.)                                                        //
//     The algorithm follows:                                                 //
//                  x[n-1] = B[n-1]/U[n-1][n-1], and                          //
//     x[i] = [B[i] - (U[i][i+1] * x[i+1]  + ... + U[i][n-1] * x[n-1])]       //
//                                                                 / U[i][i], //
//     for i = n-2, ..., 0.                                                   //
//                                                                            //
//  Arguments:                                                                //
//     double *U   Pointer to the first element of the upper triangular       //
//                 matrix.                                                    //
//     double *B   Pointer to the column vector, (n x 1) matrix, B.           //
//     double *x   Pointer to the column vector, (n x 1) matrix, x.           //
//     int     n   The number of rows or columns of the matrix U.             //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix U is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N], B[N], x[N];                                            //
//                                                                            //
//     (your code to create matrix A and column vector B)                     //
//     err = Upper_Triangular_Solve(&A[0][0], B, x, n);                       //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else printf(" The solution is \n");                                    //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
static int Upper_Triangular_Solve(double *U, double B[], double x[], int n)
{
   int i, k;

//         Solve the linear equation Ux = B for x, where U is an upper
//         triangular matrix.                                      
   
   for (k = n-1, U += n * (n - 1); k >= 0; U -= n, k--) {
      if (*(U + k) == 0.0) return -1;           // The matrix U is singular
      x[k] = B[k];
      for (i = k + 1; i < n;++i)x[k] -= x[i] * *(U + i);
      x[k] /= *(U + k);
   }

   return 0;
}


////////////////////////////////////////////////////////////////////////////////
//  int Upper_Triangular_Inverse(double *U,  int n)                           //
//                                                                            //
//  Description:                                                              //
//     This routine calculates the inverse of the upper triangular matrix U.  //
//     The subdiagonal part of the matrix is not addressed.                   //
//     The algorithm follows:                                                 //
//        Let M be the inverse of U, then U M = I,                            //
//     M[n-1][n-1] = 1.0 / U[n-1][n-1] and                                    //
//     M[i][j] = -( U[i][i+1] M[i+1][j] + ... + U[i][j] M[j][j] ) / U[i][i],  //
//     for i = n-2, ... , 0,  j = n-1, ..., i+1.                              //
//                                                                            //
//                                                                            //
//  Arguments:                                                                //
//     double *U   On input, the pointer to the first element of the matrix   //
//                 whose upper triangular elements form the matrix which is   //
//                 to be inverted. On output, the upper triangular part is    //
//                 replaced by the inverse.  The subdiagonal elements are     //
//                 not modified.                                              //
//     int     n   The number of rows and/or columns of the matrix U.         //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix U is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double U[N][N];                                                        //
//                                                                            //
//     (your code to create the matrix U)                                     //
//     err = Upper_Triangular_Inverse(&U[0][0], N);                           //
//     if (err < 0) printf(" Matrix U is singular\n");                        //
//     else {                                                                 //
//        printf(" The inverse is \n");                                       //
//           ...                                                              //
//     }                                                                      //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
static int Upper_Triangular_Inverse(double *U, int n)
{
   int i, j, k;
   double *p_i, *p_j, *p_k;
   double sum;

//         Invert the diagonal elements of the upper triangular matrix U.

   for (k = 0, p_k = U; k < n; p_k += (n + 1),++k) {
      if (*p_k == 0.0) return -1;
      else *p_k = 1.0 / *p_k;
   }

//         Invert the remaining upper triangular matrix U.

   for (i = n - 2, p_i = U + n * (n - 2); i >=0; p_i -= n, i-- ) {
      for (j = n - 1; j > i; j--) {
         sum = 0.0;
         for (k = i + 1, p_k = p_i + n; k <= j; p_k += n,++k ) {
            sum += *(p_i + k) * *(p_k + j);
         }
         *(p_i + j) = - *(p_i + i) * sum;
      }
   }
  
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
//  int Choleski_LU_Decomposition(double *A, int n)                           //
//                                                                            //
//  Description:                                                              //
//     This routine uses Choleski's method to decompose the n x n positive    //
//     definite symmetric matrix A into the product of a lower triangular     //
//     matrix L and an upper triangular matrix U equal to the transpose of L. //
//     The original matrix A is replaced by L and U with L stored in the      //
//     lower triangular part of A and the transpose U in the upper triangular //
//     part of A. The original matrix A is therefore destroyed.               //
//                                                                            //
//     Choleski's decomposition is performed by evaluating, in order, the     //
//     following pair of expressions for k = 0, ... ,n-1 :                    //
//       L[k][k] = sqrt( A[k][k] - ( L[k][0] ^ 2 + ... + L[k][k-1] ^ 2 ) )    //
//       L[i][k] = (A[i][k] - (L[i][0]*L[k][0] + ... + L[i][k-1]*L[k][k-1]))  //
//                          / L[k][k]                                         //
//     and subsequently setting                                               //
//       U[k][i] = L[i][k], for i = k+1, ... , n-1.                           //
//                                                                            //
//     After performing the LU decomposition for A, call Choleski_LU_Solve    //
//     to solve the equation Ax = B or call Choleski_LU_Inverse to calculate  //
//     the inverse of A.                                                      //
//                                                                            //
//  Arguments:                                                                //
//     double *A   On input, the pointer to the first element of the matrix   //
//                 A[n][n].  On output, the matrix A is replaced by the lower //
//                 and upper triangular Choleski factorizations of A.         //
//     int     n   The number of rows and/or columns of the matrix A.         //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix A is not positive definite symmetric (within   //
//                  working accuracy).                                        //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N];                                                        //
//                                                                            //
//     (your code to initialize the matrix A)                                 //
//     err = Choleski_LU_Decomposition((double *) A, N);                      //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else { printf(" The LLt decomposition of A is \n");                    //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Choleski_LU_Decomposition(double *A, int n)
{
   int i, k, p;
   double *p_Lk0;                   // pointer to L[k][0]
   double *p_Lkp;                   // pointer to L[k][p]  
   double *p_Lkk;                   // pointer to diagonal element on row k.
   double *p_Li0;                   // pointer to L[i][0]
   double reciprocal;

   for (k = 0, p_Lk0 = A; k < n; p_Lk0 += n,++k) {
           
//            Update pointer to row k diagonal element.   

      p_Lkk = p_Lk0 + k;

//            Calculate the difference of the diagonal element in row k
//            from the sum of squares of elements row k from column 0 to 
//            column k-1.

      for (p = 0, p_Lkp = p_Lk0; p < k; p_Lkp += 1,  p++)
         *p_Lkk -= *p_Lkp * *p_Lkp;

//            If diagonal element is not positive, return the error code,
//            the matrix is not positive definite symmetric.

      if ( *p_Lkk <= 0.0 ) return -1;

//            Otherwise take the square root of the diagonal element.

      *p_Lkk = sqrt( *p_Lkk );
      reciprocal = 1.0 / *p_Lkk;

//            For rows i = k+1 to n-1, column k, calculate the difference
//            between the i,k th element and the inner product of the first
//            k-1 columns of row i and row k, then divide the difference by
//            the diagonal element in row k.
//            Store the transposed element in the upper triangular matrix.

      p_Li0 = p_Lk0 + n;
      for (i = k + 1; i < n; p_Li0 += n,++i){
         for (p = 0; p < k; p++)
            *(p_Li0 + k) -= *(p_Li0 + p) * *(p_Lk0 + p);
         *(p_Li0 + k) *= reciprocal;
         *(p_Lk0 + i) = *(p_Li0 + k);
      }  
   }
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
//  int Choleski_LU_Solve(double *LU, double *B, double *x,  int n)           //
//                                                                            //
//  Description:                                                              //
//     This routine uses Choleski's method to solve the linear equation       //
//     Ax = B.  This routine is called after the matrix A has been decomposed //
//     into a product of a lower triangular matrix L and an upper triangular  //
//     matrix U which is the transpose of L. The matrix A is the product LU.  //
//     The solution proceeds by solving the linear equation Ly = B for y and  //
//     subsequently solving the linear equation Ux = y for x.                 //
//                                                                            //
//  Arguments:                                                                //
//     double *LU  Pointer to the first element of the matrix whose elements  //
//                 form the lower and upper triangular matrix factors of A.   //
//     double *B   Pointer to the column vector, (n x 1) matrix, B            //
//     double *x   Solution to the equation Ax = B.                           //
//     int     n   The number of rows and/or columns of the matrix LU.        //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix L is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N], B[N], x[N];                                            //
//                                                                            //
//     (your code to create matrix A and column vector B)                     //
//     err = Choleski_LU_Decomposition(&A[0][0], N);                          //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else {                                                                 //
//        err = Choleski_LU_Solve(&A[0][0], B, x, n);                         //
//        if (err < 0) printf(" Matrix A is singular\n");                     //
//        else printf(" The solution is \n");                                 //
//           ...                                                              //
//     }                                                                      //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Choleski_LU_Solve(double *LU, double B[], double x[], int n)
{

//         Solve the linear equation Ly = B for y, where L is a lower
//         triangular matrix.
   
   if ( Lower_Triangular_Solve(LU, B, x, n) < 0 ) return -1;

//         Solve the linear equation Ux = y, where y is the solution
//         obtained above of Ly = B and U is an upper triangular matrix.

   return Upper_Triangular_Solve(LU, x, x, n);
}


////////////////////////////////////////////////////////////////////////////////
//  int Choleski_LU_Inverse(double *LU,  int n)                               //
//                                                                            //
//  Description:                                                              //
//     This routine uses Choleski's method to find the inverse of the matrix  //
//     A.  This routine is called after the matrix A has been decomposed      //
//     into a product of a lower triangular matrix L and an upper triangular  //
//     matrix U which is the transpose of L. The matrix A is the product of   //
//     the L and U.  Upon completion, the inverse of A is stored in LU so     //
//     that the matrix LU is destroyed.                                       //
//                                                                            //
//  Arguments:                                                                //
//     double *LU  On input, the pointer to the first element of the matrix   //
//                 whose elements form the lower and upper triangular matrix  //
//                 factors of A.  On output, the matrix LU is replaced by the //
//                 inverse of the matrix A equal to the product of L and U.   //
//     int     n   The number of rows and/or columns of the matrix LU.        //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix L is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N], B[N], x[N];                                            //
//                                                                            //
//     (your code to create matrix A and column vector B)                     //
//     err = Choleski_LU_Decomposition(&A[0][0], N);                          //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else {                                                                 //
//        err = Choleski_LU_Inverse(&A[0][0], n);                             //
//        if (err < 0) printf(" Matrix A is singular\n");                     //
//        else printf(" The inverse is \n");                                  //
//           ...                                                              //
//     }                                                                      //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Choleski_LU_Inverse(double *LU, int n)
{
   int i, j, k;
   double *p_i, *p_j, *p_k;
   double sum;

   if ( Lower_Triangular_Inverse(LU, n) < 0 ) return -1;
  
//         Premultiply L inverse by the transpose of L inverse.      

   for (i = 0, p_i = LU; i < n;++i, p_i += n) {
      for (j = 0, p_j = LU; j <= i; j++, p_j += n) {
         sum = 0.0;
         for (k = i, p_k = p_i; k < n;++k, p_k += n)
            sum += *(p_k + i) * *(p_k + j);
         *(p_i + j) = sum;
         *(p_j + i) = sum;
      }
   }
  
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
// File: singular_value_decomposition.c                                       //
// Contents:                                                                  //
//    Singular_Value_Decomposition                                            //
//    Singular_Value_Decomposition_Solve                                      //
//    Singular_Value_Decomposition_Inverse                                    //
////////////////////////////////////////////////////////////////////////////////

#include <string.h>              // required for memcpy()
#include <float.h>               // required for DBL_EPSILON
#include <math.h>                // required for fabs(), sqrt();

#define MAX_ITERATION_COUNT 30   // Maximum number of iterations

//                        Internally Defined Routines 
static void Householders_Reduction_to_Bidiagonal_Form(double* A, int nrows,
    int ncols, double* U, double* V, double* diagonal, double* superdiagonal );
static int  Givens_Reduction_to_Diagonal_Form( int nrows, int ncols,
           double* U, double* V, double* diagonal, double* superdiagonal );
static void Sort_by_Decreasing_Singular_Values(int nrows, int ncols,
                                double* singular_value, double* U, double* V);

////////////////////////////////////////////////////////////////////////////////
//  int Singular_Value_Decomposition(double* A, int nrows, int ncols,         //
//        double* U, double* singular_values, double* V, double* dummy_array) //
//                                                                            //
//  Description:                                                              //
//     This routine decomposes an m x n matrix A, with m >= n, into a product //
//     of the three matrices U, D, and V', i.e. A = UDV', where U is an m x n //
//     matrix whose columns are orthogonal, D is a n x n diagonal matrix, and //
//     V is an n x n orthogonal matrix.  V' denotes the transpose of V.  If   //
//     m < n, then the procedure may be used for the matrix A'.  The singular //
//     values of A are the diagonal elements of the diagonal matrix D and     //
//     correspond to the positive square roots of the eigenvalues of the      //
//     matrix A'A.                                                            //
//                                                                            //
//     This procedure programmed here is based on the method of Golub and     //
//     Reinsch as given on pages 134 - 151 of the "Handbook for Automatic     //
//     Computation vol II - Linear Algebra" edited by Wilkinson and Reinsch   //
//     and published by Springer-Verlag, 1971.                                //
//                                                                            //
//     The Golub and Reinsch's method for decomposing the matrix A into the   //
//     product U, D, and V' is performed in three stages:                     //
//       Stage 1:  Decompose A into the product of three matrices U1, B, V1'  //
//         A = U1 B V1' where B is a bidiagonal matrix, and U1, and V1 are a  //
//         product of Householder transformations.                            //
//       Stage 2:  Use Given' transformations to reduce the bidiagonal matrix //
//         B into the product of the three matrices U2, D, V2'.  The singular //
//         value decomposition is then UDV'where U = U2 U1 and V' = V1' V2'.  //
//       Stage 3:  Sort the matrix D in decreasing order of the singular      //
//         values and interchange the columns of both U and V to reflect any  //
//         change in the order of the singular values.                        //
//                                                                            //
//     After performing the singular value decomposition for A, call          //
//     Singular_Value_Decomposition to solve the equation Ax = B or call      //
//     Singular_Value_Decomposition_Inverse to calculate the pseudo-inverse   //
//     of A.                                                                  //
//                                                                            //
//  Arguments:                                                                //
//     double* A                                                              //
//        On input, the pointer to the first element of the matrix            //
//        A[nrows][ncols].  The matrix A is unchanged.                        //
//     int nrows                                                              //
//        The number of rows of the matrix A.                                 //
//     int ncols                                                              //
//        The number of columns of the matrix A.                              //
//     double* U                                                              //
//        On input, a pointer to a matrix with the same number of rows and    //
//        columns as the matrix A.  On output, the matrix with mutually       //
//        orthogonal columns which is the left-most factor in the singular    //
//        value decomposition of A.                                           //
//     double* singular_values                                                //
//        On input, a pointer to an array dimensioned to same as the number   //
//        of columns of the matrix A, ncols.  On output, the singular values  //
//        of the matrix A sorted in decreasing order.  This array corresponds //
//        to the diagonal matrix in the singular value decomposition of A.    //
//     double* V                                                              //
//        On input, a pointer to a square matrix with the same number of rows //
//        and columns as the columns of the matrix A, i.e. V[ncols][ncols].   //
//        On output, the orthogonal matrix whose transpose is the right-most  //
//        factor in the singular value decomposition of A.                    //
//     double* dummy_array                                                    //
//        On input, a pointer to an array dimensioned to same as the number   //
//        of columns of the matrix A, ncols.  This array is used to store     //
//        the super-diagonal elements resulting from the Householder reduction//
//        of the matrix A to bidiagonal form.  And as an input to the Given's //
//        procedure to reduce the bidiagonal form to diagonal form.           //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - During the Given's reduction of the bidiagonal form to    //
//                  diagonal form the procedure failed to terminate within    //
//                  MAX_ITERATION_COUNT iterations.                           //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double A[M][N];                                                        //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double singular_values[N];                                             //
//     double* dummy_array;                                                   //
//                                                                            //
//     (your code to initialize the matrix A)                                 //
//     dummy_array = (double*) malloc(N * sizeof(double));                    //
//     if (dummy_array == NULL) {printf(" No memory available\n"); exit(0); } //
//                                                                            //
//     err = Singular_Value_Decomposition((double*) A, M, N, (double*) U,     //
//                              singular_values, (double*) V, dummy_array);   //
//                                                                            //
//     free(dummy_array);                                                     //
//     if (err < 0) printf(" Failed to converge\n");                          //
//     else { printf(" The singular value decomposition of A is \n");         //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Singular_Value_Decomposition(double* A, int nrows, int ncols, double* U, 
                      double* singular_values, double* V, double* dummy_array)
{
   Householders_Reduction_to_Bidiagonal_Form( A, nrows, ncols, U, V,
                                                singular_values, dummy_array);

   if (Givens_Reduction_to_Diagonal_Form( nrows, ncols, U, V,
                                singular_values, dummy_array ) < 0) return -1;

   Sort_by_Decreasing_Singular_Values(nrows, ncols, singular_values, U, V);
  
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
// static void Householders_Reduction_to_Bidiagonal_Form(double* A, int nrows,//
//  int ncols, double* U, double* V, double* diagonal, double* superdiagonal )//
//                                                                            //
//  Description:                                                              //
//     This routine decomposes an m x n matrix A, with m >= n, into a product //
//     of the three matrices U, B, and V', i.e. A = UBV', where U is an m x n //
//     matrix whose columns are orthogonal, B is a n x n bidiagonal matrix,   //
//     and V is an n x n orthogonal matrix.  V' denotes the transpose of V.   //
//     If m < n, then the procedure may be used for the matrix A'.  The       //
//                                                                            //
//     The matrix U is the product of Householder transformations which       //
//     annihilate the subdiagonal components of A while the matrix V is       //
//     the product of Householder transformations which annihilate the        //
//     components of A to the right of the superdiagonal.                     //
//                                                                            //
//     The Householder transformation which leaves invariant the first k-1    //
//     elements of the k-th column and annihilates the all the elements below //
//     the diagonal element is P = I - (2/u'u)uu', u is an nrows-dimensional  //
//     vector the first k-1 components of which are zero and the last         //
//     components agree with the current transformed matrix below the diagonal//
//     diagonal, the remaining k-th element is the diagonal element - s, where//
//     s = (+/-)sqrt(sum of squares of the elements below the diagonal), the  //
//     sign is chosen opposite that of the diagonal element.                  //
//                                                                            //
//  Arguments:                                                                //
//     double* A                                                              //
//        On input, the pointer to the first element of the matrix            //
//        A[nrows][ncols].  The matrix A is unchanged.                        //
//     int nrows                                                              //
//        The number of rows of the matrix A.                                 //
//     int ncols                                                              //
//        The number of columns of the matrix A.                              //
//     double* U                                                              //
//        On input, a pointer to a matrix with the same number of rows and    //
//        columns as the matrix A.  On output, the matrix with mutually       //
//        orthogonal columns which is the left-most factor in the bidiagonal  //
//        decomposition of A.                                                 //
//     double* V                                                              //
//        On input, a pointer to a square matrix with the same number of rows //
//        and columns as the columns of the matrix A, i.e. V[ncols][ncols].   //
//        On output, the orthogonal matrix whose transpose is the right-most  //
//        factor in the bidiagonal decomposition of A.                        //
//     double* diagonal                                                       //
//        On input, a pointer to an array dimensioned to same as the number   //
//        of columns of the matrix A, ncols.  On output, the diagonal of the  //
//        bidiagonal matrix.                                                  //
//     double* superdiagonal                                                  //
//        On input, a pointer to an array dimensioned to same as the number   //
//        of columns of the matrix A, ncols.  On output, the superdiagonal    //
//        of the bidiagonal matrix.                                           //
//                                                                            //
//  Return Values:                                                            //
//     The function is of type void and therefore does not return a value.    //
//     The matrices U, V, and the diagonal and superdiagonal are calculated   //
//     using the addresses passed in the argument list.                       //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double A[M][N];                                                        //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double diagonal[N];                                                    //
//     double superdiagonal[N];                                               //
//                                                                            //
//     (your code to initialize the matrix A - Note this routine is not       //
//     (accessible from outside i.e. it is declared static)                   //
//                                                                            //
//     Householders_Reduction_to_Bidiagonal_Form((double*) A, nrows, ncols,   //
//                   (double*) U, (double*) V, diagonal, superdiagonal )      //
//                                                                            //
//     free(dummy_array);                                                     //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
static void Householders_Reduction_to_Bidiagonal_Form(double* A, int nrows,
    int ncols, double* U, double* V, double* diagonal, double* superdiagonal )
{
   int i,j,k,ip1;
   double s, s2, si, scale;
   double dum;
   double *pu, *pui, *pv, *pvi;
   double half_norm_squared;

// Copy A to U

   memcpy(U,A, sizeof(double) * nrows * ncols);

//
 
   diagonal[0] = 0.0;
   s = 0.0;
   scale = 0.0;
   for ( i = 0, pui = U, ip1 = 1; i < ncols; pui += ncols,++i, ip1++ ) {
      superdiagonal[i] = scale * s;
//       
//                  Perform Householder transform on columns.
//
//       Calculate the normed squared of the i-th column vector starting at 
//       row i.
//
      for (j = i, pu = pui, scale = 0.0; j < nrows; j++, pu += ncols)
         scale += fabs( *(pu + i) );
       
      if (scale > 0.0) {
         for (j = i, pu = pui, s2 = 0.0; j < nrows; j++, pu += ncols) {
            *(pu + i) /= scale;
            s2 += *(pu + i) * *(pu + i);
         }
//
//    
//       Chose sign of s which maximizes the norm
//  
         s = ( *(pui + i) < 0.0 ) ? sqrt(s2) : -sqrt(s2);
//
//       Calculate -2/u'u
//
         half_norm_squared = *(pui + i) * s - s2;
//
//       Transform remaining columns by the Householder transform.
//
         *(pui + i) -= s;
         
         for (j = ip1; j < ncols; ++j) {
            for (k = i, si = 0.0, pu = pui; k < nrows;++k, pu += ncols)
               si += *(pu + i) * *(pu + j);
            si /= half_norm_squared;
            for (k = i, pu = pui; k < nrows;++k, pu += ncols) {
               *(pu + j) += si * *(pu + i);
            }
         }
      }
      for (j = i, pu = pui; j < nrows; j++, pu += ncols) *(pu + i) *= scale;
      diagonal[i] = s * scale;
//       
//                  Perform Householder transform on rows.
//
//       Calculate the normed squared of the i-th row vector starting at 
//       column i.
//
      s = 0.0;
      scale = 0.0;
      if (i >= nrows || i == (ncols - 1) ) continue;
      for (j = ip1; j < ncols; ++j) scale += fabs ( *(pui + j) );
      if ( scale > 0.0 ) {
         for (j = ip1, s2 = 0.0; j < ncols; ++j) {
            *(pui + j) /= scale;
            s2 += *(pui + j) * *(pui + j);
         }
         s = ( *(pui + ip1) < 0.0 ) ? sqrt(s2) : -sqrt(s2);
//
//       Calculate -2/u'u
//
         half_norm_squared = *(pui + ip1) * s - s2;
//
//       Transform the rows by the Householder transform.
//
         *(pui + ip1) -= s;
         for (k = ip1; k < ncols;++k)
            superdiagonal[k] = *(pui + k) / half_norm_squared;
         if ( i < (nrows - 1) ) {
            for (j = ip1, pu = pui + ncols; j < nrows; j++, pu += ncols) {
               for (k = ip1, si = 0.0; k < ncols;++k) 
                  si += *(pui + k) * *(pu + k);
               for (k = ip1; k < ncols;++k) { 
                  *(pu + k) += si * superdiagonal[k];
               }
            }
         }
         for (k = ip1; k < ncols;++k) *(pui + k) *= scale;
      }
   }

// Update V
   pui = U + ncols * (ncols - 2);
   pvi = V + ncols * (ncols - 1);
   *(pvi + ncols - 1) = 1.0;
   s = superdiagonal[ncols - 1];
   pvi -= ncols;
   for (i = ncols - 2, ip1 = ncols - 1; i >= 0; i--, pui -= ncols,
                                                      pvi -= ncols, ip1-- ) {
      if ( s != 0.0 ) {
         pv = pvi + ncols;
         for (j = ip1; j < ncols; j++, pv += ncols)
            *(pv + i) = ( *(pui + j) / *(pui + ip1) ) / s;
         for (j = ip1; j < ncols; ++j) { 
            si = 0.0;
            for (k = ip1, pv = pvi + ncols; k < ncols;++k, pv += ncols)
               si += *(pui + k) * *(pv + j);
            for (k = ip1, pv = pvi + ncols; k < ncols;++k, pv += ncols)
               *(pv + j) += si * *(pv + i);                  
         }
      }
      pv = pvi + ncols;
      for ( j = ip1; j < ncols; j++, pv += ncols ) {
         *(pvi + j) = 0.0;
         *(pv + i) = 0.0;
      }
      *(pvi + i) = 1.0;
      s = superdiagonal[i];
   }

// Update U

   pui = U + ncols * (ncols - 1);
   for (i = ncols - 1, ip1 = ncols; i >= 0; ip1 = i, i--, pui -= ncols ) {
      s = diagonal[i];
      for ( j = ip1; j < ncols; ++j) *(pui + j) = 0.0;
      if ( s != 0.0 ) {
         for (j = ip1; j < ncols; ++j) { 
            si = 0.0;
            pu = pui + ncols;
            for (k = ip1; k < nrows;++k, pu += ncols)
               si += *(pu + i) * *(pu + j);
            si = (si / *(pui + i) ) / s;
            for (k = i, pu = pui; k < nrows;++k, pu += ncols)
               *(pu + j) += si * *(pu + i);                  
         }
         for (j = i, pu = pui; j < nrows; j++, pu += ncols){
            *(pu + i) /= s;
         }
      }
      else 
         for (j = i, pu = pui; j < nrows; j++, pu += ncols) *(pu + i) = 0.0;
      *(pui + i) += 1.0;
   }
}


////////////////////////////////////////////////////////////////////////////////
// static int Givens_Reduction_to_Diagonal_Form( int nrows, int ncols,        //
//         double* U, double* V, double* diagonal, double* superdiagonal )    //
//                                                                            //
//  Description:                                                              //
//     This routine decomposes a bidiagonal matrix given by the arrays        //
//     diagonal and superdiagonal into a product of three matrices U1, D and  //
//     V1', the matrix U1 premultiplies U and is returned in U, the matrix    //
//     V1 premultiplies V and is returned in V.  The matrix D is a diagonal   //
//     matrix and replaces the array diagonal.                                //
//                                                                            //
//     The method used to annihilate the offdiagonal elements is a variant    //
//     of the QR transformation.  The method consists of applying Givens      //
//     rotations to the right and the left of the current matrix until        //
//     the new off-diagonal elements are chased out of the matrix.            //
//                                                                            //
//     The process is an iterative process which due to roundoff errors may   //
//     not converge within a predefined number of iterations.  (This should   //
//     be unusual.)                                                           //
//                                                                            //
//  Arguments:                                                                //
//     int nrows                                                              //
//        The number of rows of the matrix U.                                 //
//     int ncols                                                              //
//        The number of columns of the matrix U.                              //
//     double* U                                                              //
//        On input, a pointer to a matrix already initialized to a matrix     //
//        with mutually orthogonal columns.   On output, the matrix with      //
//        mutually orthogonal columns.                                        //
//     double* V                                                              //
//        On input, a pointer to a square matrix with the same number of rows //
//        and columns as the columns of the matrix U, i.e. V[ncols][ncols].   //
//        The matrix V is assumed to be initialized to an orthogonal matrix.  //
//        On output, V is an orthogonal matrix.                               //
//     double* diagonal                                                       //
//        On input, a pointer to an array of dimension ncols which initially  //
//        contains the diagonal of the bidiagonal matrix.  On output, the     //
//        it contains the diagonal of the diagonal matrix.                    //
//     double* superdiagonal                                                  //
//        On input, a pointer to an array of dimension ncols which initially  //
//        the first component is zero and the successive components form the  //
//        superdiagonal of the bidiagonal matrix.                             //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The procedure failed to terminate within                  //
//                  MAX_ITERATION_COUNT iterations.                           //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double diagonal[N];                                                    //
//     double superdiagonal[N];                                               //
//     int err;                                                               //
//                                                                            //
//     (your code to initialize the matrices U, V, diagonal, and )            //
//     ( superdiagonal.  - Note this routine is not accessible from outside)  //
//     ( i.e. it is declared static.)                                         //
//                                                                            //
//     err = Givens_Reduction_to_Diagonal_Form( M,N,(double*)U,(double*)V,    //
//                                                 diagonal, superdiagonal ); //
//     if ( err < 0 ) printf("Failed to converge\n");                         //
//     else { ... }                                                           //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
static int Givens_Reduction_to_Diagonal_Form( int nrows, int ncols,
           double* U, double* V, double* diagonal, double* superdiagonal )
{

   double epsilon;
   double c, s;
   double f,g,h;
   double x,y,z;
   double *pu, *pv;
   int i,j,k,m;
   int rotation_test;
   int iteration_count;
  
   for (i = 0, x = 0.0; i < ncols;++i){
      y = fabs(diagonal[i]) + fabs(superdiagonal[i]);
      if ( x < y ) x = y;
   }
   epsilon = x * DBL_EPSILON;
   for (k = ncols - 1; k >= 0; k--) {
      iteration_count = 0;
      while(1) {
         rotation_test = 1;
         for (m = k; m >= 0; m--) { 
            if (fabs(superdiagonal[m]) <= epsilon) {rotation_test = 0; break;}
            if (fabs(diagonal[m-1]) <= epsilon) break;
         }
         if (rotation_test) {
            c = 0.0;
            s = 1.0;
            for (i = m; i <= k;++i){  
               f = s * superdiagonal[i];
               superdiagonal[i] *= c;
               if (fabs(f) <= epsilon) break;
               g = diagonal[i];
               h = sqrt(f*f + g*g);
               diagonal[i] = h;
               c = g / h;
               s = -f / h; 
               for (j = 0, pu = U; j < nrows; j++, pu += ncols) { 
                  y = *(pu + m - 1);
                  z = *(pu + i);
                  *(pu + m - 1 ) = y * c + z * s;
                  *(pu + i) = -y * s + z * c;
               }
            }
         }
         z = diagonal[k];
         if (m == k ) {
            if ( z < 0.0 ) {
               diagonal[k] = -z;
               for ( j = 0, pv = V; j < ncols; j++, pv += ncols) 
                  *(pv + k) = - *(pv + k);
            }
            break;
         }
         else {
            if ( iteration_count >= MAX_ITERATION_COUNT ) return -1;
            iteration_count++;
            x = diagonal[m];
            y = diagonal[k-1];
            g = superdiagonal[k-1];
            h = superdiagonal[k];
            f = ( (y - z) * ( y + z ) + (g - h) * (g + h) )/(2.0 * h * y);
            g = sqrt( f * f + 1.0 );
            if ( f < 0.0 ) g = -g;
            f = ( (x - z) * (x + z) + h * (y / (f + g) - h) ) / x;
// Next QR Transformtion
            c = 1.0;
            s = 1.0;
            for (i = m + 1; i <= k;++i){
               g = superdiagonal[i];
               y = diagonal[i];
               h = s * g;
               g *= c;
               z = sqrt( f * f + h * h );
               superdiagonal[i-1] = z;
               c = f / z;
               s = h / z;
               f =  x * c + g * s;
               g = -x * s + g * c;
               h = y * s;
               y *= c;
               for (j = 0, pv = V; j < ncols; j++, pv += ncols) {
                  x = *(pv + i - 1);
                  z = *(pv + i);
                  *(pv + i - 1) = x * c + z * s;
                  *(pv + i) = -x * s + z * c;
               }
               z = sqrt( f * f + h * h );
               diagonal[i - 1] = z;
               if (z != 0.0) {
                  c = f / z;
                  s = h / z;
               } 
               f = c * g + s * y;
               x = -s * g + c * y;
               for (j = 0, pu = U; j < nrows; j++, pu += ncols) {
                  y = *(pu + i - 1);
                  z = *(pu + i);
                  *(pu + i - 1) = c * y + s * z;
                  *(pu + i) = -s * y + c * z;
               }
            }
            superdiagonal[m] = 0.0;
            superdiagonal[k] = f;
            diagonal[k] = x;
         }
      } 
   }
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
// static void Sort_by_Decreasing_Singular_Values(int nrows, int ncols,       //
//                            double* singular_values, double* U, double* V)  //
//                                                                            //
//  Description:                                                              //
//     This routine sorts the singular values from largest to smallest        //
//     singular value and interchanges the columns of U and the columns of V  //
//     whenever a swap is made.  I.e. if the i-th singular value is swapped   //
//     with the j-th singular value, then the i-th and j-th columns of U are  //
//     interchanged and the i-th and j-th columns of V are interchanged.      //
//                                                                            //
//  Arguments:                                                                //
//     int nrows                                                              //
//        The number of rows of the matrix U.                                 //
//     int ncols                                                              //
//        The number of columns of the matrix U.                              //
//     double* singular_values                                                //
//        On input, a pointer to the array of singular values.  On output, the//
//        sorted array of singular values.                                    //
//     double* U                                                              //
//        On input, a pointer to a matrix already initialized to a matrix     //
//        with mutually orthogonal columns.  On output, the matrix with       //
//        mutually orthogonal possibly permuted columns.                      //
//     double* V                                                              //
//        On input, a pointer to a square matrix with the same number of rows //
//        and columns as the columns of the matrix U, i.e. V[ncols][ncols].   //
//        The matrix V is assumed to be initialized to an orthogonal matrix.  //
//        On output, V is an orthogonal matrix with possibly permuted columns.//
//                                                                            //
//  Return Values:                                                            //
//        The function is of type void.                                       //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double diagonal[N];                                                    //
//                                                                            //
//     (your code to initialize the matrices U, V, and diagonal. )            //
//     ( - Note this routine is not accessible from outside)                  //
//     ( i.e. it is declared static.)                                         //
//                                                                            //
//     Sort_by_Decreasing_Singular_Values(nrows, ncols, singular_values,      //
//                                                 (double*) U, (double*) V); //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
static void Sort_by_Decreasing_Singular_Values(int nrows, int ncols,
                                double* singular_values, double* U, double* V)
{
   int i,j,max_index;
   double temp;
   double *p1, *p2;

   for (i = 0; i < ncols - 1;++i){
      max_index = i;
      for (j = i + 1; j < ncols; ++j)
         if (singular_values[j] > singular_values[max_index] ) 
            max_index = j;
      if (max_index == i) continue;
      temp = singular_values[i];
      singular_values[i] = singular_values[max_index];
      singular_values[max_index] = temp;
      p1 = U + max_index;
      p2 = U + i;
      for (j = 0; j < nrows; j++, p1 += ncols, p2 += ncols) {
         temp = *p1;
         *p1 = *p2;
         *p2 = temp;
      } 
      p1 = V + max_index;
      p2 = V + i;
      for (j = 0; j < ncols; j++, p1 += ncols, p2 += ncols) {
         temp = *p1;
         *p1 = *p2;
         *p2 = temp;
      }
   } 
}


////////////////////////////////////////////////////////////////////////////////
//  void Singular_Value_Decomposition_Solve(double* U, double* D, double* V,  //
//              double tolerance, int nrows, int ncols, double *B, double* x) //
//                                                                            //
//  Description:                                                              //
//     This routine solves the system of linear equations Ax=B where A =UDV', //
//     is the singular value decomposition of A.  Given UDV'x=B, then         //
//     x = V(1/D)U'B, where 1/D is the pseudo-inverse of D, i.e. if D[i] > 0  //
//     then (1/D)[i] = 1/D[i] and if D[i] = 0, then (1/D)[i] = 0.  Since      //
//     the singular values are subject to round-off error.  A tolerance is    //
//     given so that if D[i] < tolerance, D[i] is treated as if it is 0.      //
//     The default tolerance is D[0] * DBL_EPSILON * ncols, if the user       //
//     specified tolerance is less than the default tolerance, the default    //
//     tolerance is used.                                                     //
//                                                                            //
//  Arguments:                                                                //
//     double* U                                                              //
//        A matrix with mutually orthonormal columns.                         //
//     double* D                                                              //
//        A diagonal matrix with decreasing non-negative diagonal elements.   //
//        i.e. D[i] > D[j] if i < j and D[i] >= 0 for all i.                  //
//     double* V                                                              //
//        An orthogonal matrix.                                               //
//     double tolerance                                                       //
//        An lower bound for non-zero singular values (provided tolerance >   //
//        ncols * DBL_EPSILON * D[0]).                                        //
//     int nrows                                                              //
//        The number of rows of the matrix U and B.                           //
//     int ncols                                                              //
//        The number of columns of the matrix U.  Also the number of rows and //
//        columns of the matrices D and V.                                    //
//     double* B                                                              //
//        A pointer to a vector dimensioned as nrows which is the  right-hand //
//        side of the equation Ax = B where A = UDV'.                         //
//     double* x                                                              //
//        A pointer to a vector dimensioned as ncols, which is the least      //
//        squares solution of the equation Ax = B where A = UDV'.             //
//                                                                            //
//  Return Values:                                                            //
//        The function is of type void.                                       //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     #define NB                                                             //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double D[N];                                                           //
//     double B[M];                                                           //
//     double x[N];                                                           //
//     double tolerance;                                                      //
//                                                                            //
//     (your code to initialize the matrices U,D,V,B)                         //
//                                                                            //
//     Singular_Value_Decomposition_Solve((double*) U, D, (double*) V,        //
//                                              tolerance, M, N, B, x, bcols) //
//                                                                            //
//     printf(" The solution of Ax=B is \n");                                 //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //

void Singular_Value_Decomposition_Solve(double* U, double* D, double* V,  
                double tolerance, int nrows, int ncols, double *B, double* x) 
{
   int i,j,k;
   double *pu, *pv;
   double dum;

   dum = DBL_EPSILON * D[0] * (double) ncols;
   if (tolerance < dum) tolerance = dum;

   for ( i = 0, pv = V; i < ncols;++i, pv += ncols) {
      x[i] = 0.0;
      for (j = 0; j < ncols; ++j)
         if (D[j] > tolerance ) {
            for (k = 0, dum = 0.0, pu = U; k < nrows;++k, pu += ncols)
               dum += *(pu + j) * B[k];
            x[i] += dum * *(pv + j) / D[j];
         }
   } 
}


////////////////////////////////////////////////////////////////////////////////
//  void Singular_Value_Decomposition_Inverse(double* U, double* D, double* V,//
//                     double tolerance, int nrows, int ncols, double *Astar) //
//                                                                            //
//  Description:                                                              //
//     This routine calculates the pseudo-inverse of the matrix A = UDV'.     //
//     where U, D, V constitute the singular value decomposition of A.        //
//     Let Astar be the pseudo-inverse then Astar = V(1/D)U', where 1/D is    //
//     the pseudo-inverse of D, i.e. if D[i] > 0 then (1/D)[i] = 1/D[i] and   //
//     if D[i] = 0, then (1/D)[i] = 0.  Because the singular values are       //
//     subject to round-off error.  A tolerance is given so that if           //
//     D[i] < tolerance, D[i] is treated as if it were 0.                     //
//     The default tolerance is D[0] * DBL_EPSILON * ncols, assuming that the //
//     diagonal matrix of singular values is sorted from largest to smallest, //
//     if the user specified tolerance is less than the default tolerance,    //
//     then the default tolerance is used.                                    //
//                                                                            //
//  Arguments:                                                                //
//     double* U                                                              //
//        A matrix with mutually orthonormal columns.                         //
//     double* D                                                              //
//        A diagonal matrix with decreasing non-negative diagonal elements.   //
//        i.e. D[i] > D[j] if i < j and D[i] >= 0 for all i.                  //
//     double* V                                                              //
//        An orthogonal matrix.                                               //
//     double tolerance                                                       //
//        An lower bound for non-zero singular values (provided tolerance >   //
//        ncols * DBL_EPSILON * D[0]).                                        //
//     int nrows                                                              //
//        The number of rows of the matrix U and B.                           //
//     int ncols                                                              //
//        The number of columns of the matrix U.  Also the number of rows and //
//        columns of the matrices D and V.                                    //
//     double* Astar                                                          //
//        On input, a pointer to the first element of an ncols x nrows matrix.//
//        On output, the pseudo-inverse of UDV'.                              //
//                                                                            //
//  Return Values:                                                            //
//        The function is of type void.                                       //
//                                                                            //
//  Example:                                                                  //
//     #define M                                                              //
//     #define N                                                              //
//     double U[M][N];                                                        //
//     double V[N][N];                                                        //
//     double D[N];                                                           //
//     double Astar[N][M];                                                    //
//     double tolerance;                                                      //
//                                                                            //
//     (your code to initialize the matrices U,D,V)                           //
//                                                                            //
//     Singular_Value_Decomposition_Inverse((double*) U, D, (double*) V,      //
//                                        tolerance, M, N, (double*) Astar);  //
//                                                                            //
//     printf(" The pseudo-inverse of A = UDV' is \n");                       //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //

void Singular_Value_Decomposition_Inverse(double* U, double* D, double* V,  
                        double tolerance, int nrows, int ncols, double *Astar) 
{
   int i,j,k;
   double *pu, *pv, *pa;
   double dum;

   dum = DBL_EPSILON * D[0] * (double) ncols;
   if (tolerance < dum) tolerance = dum;
   for ( i = 0, pv = V, pa = Astar; i < ncols;++i, pv += ncols) 
      for ( j = 0, pu = U; j < nrows; j++, pa++) 
        for (k = 0, *pa = 0.0; k < ncols;++k, pu++)
           if (D[k] > tolerance) *pa += *(pv + k) * *pu / D[k];
}

};