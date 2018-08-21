//
// Created by Forrest Laine on 8/20/18.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl_scalapack.h>
#include <time.h>
#include "mkl.h"

#define MAX(a,b)((a)<(b)?(b):(a))
#define MIN(a,b)((a)>(b)?(b):(a))

typdef MKL_INT MDESC[ 9 ];

/* Parameters */
const double zero = 0.0E+0, one = 1.0E+0, two = 2.0E+0;
const MKL_INT i_zero = 0, i_one = 1, i_five = 5, i_negone = -1;
const char trans = 'N';


/*==== MAIN FUNCTION =================================================*/
int main(int argc, char *argv[]) {

  srand(0);
/*  ==== Declarations =================================================== */

/*  File variables */
  FILE *fin;

/*  Matrix descriptors */
  MDESC descA, descb, descx;

/*  Local scalars */
  MKL_INT iam, nprocs, ictxt_r, ictxt_c, myrow, mycol;
  MKL_INT nominal_block_size, lld;
  MKL_INT T, nx, nu, total_size, local_size;
  MKL_INT i, j, info;
  MKL_INT half_bw, band_matrix_rows, lwork, info, ipiv;

  int nx_int, nu_int, nproc_int, T_int;
  int nrows_int, ncols_int;

/*  Local arrays */
  double *A, *x, *b, *work;
  MKL_INT iwork[5];

  /*  ==== Executable statements ========================================== */

/*  Get information about how many processes are used for program execution
    and number of current process */
  blacs_pinfo_( &iam, &nprocs );

/*  Init temporary 1D process grid */
  blacs_get_( &i_negone, &i_zero, &ictxt_c );
  blacs_gridinit_( &ictxt_c, "R", &i_one, &nprocs );

/*  Open input file */
  if ( iam == 0 ) {
    fin = fopen( "lqr.in", "r" );
    if ( fin == NULL ) {
      printf( "Error while open input file." );
      return 2;
    }
  }

/*  Read data and send it to all processes */
  if ( iam == 0 ) {

/*      Read parameters */
    fscanf( fin, "%d nx, dimension of state vectors, must be > 0 ", &n_int );
    fscanf( fin, "%d nu, dimension of control vectors, must be > 0 ", &n_int );
    fscanf( fin, "%d T, trajectory length, must be > 0 ", &n_int );

//    fscanf( fin, "%d n, dimension of vectors, must be > 0 ", &n_int );
//    fscanf( fin, "%d nb, size of blocks, must be > 0 ", &nb_int );
//    fscanf( fin, "%d p, number of rows in the process grid, must be > 0", &nprow_int );
//    fscanf( fin, "%d q, number of columns in the process grid, must be > 0, p*q = number of processes", &npcol_int );
//    fscanf( fin, "%lf threshold for residual check (to switch off check set it < 0.0) ", &thresh );

    T = (MKL_INT) T_int;
    nx = (MKL_INT) nx_int;
    nu = (MKL_INT) nu_int;
    total_size = (2 * nx + nu) * (T-1) + 2 * nx;
    nominal_block_size = (MKL_INT) total_size / nprocs;

/*      Check if all parameters are correct */
    if( ( nx<nu )||( nx<=0 )||( nu<=0 )||( T<=0 )||( T <= (nx / nu)  ) ) {
      printf( "One or several input parameters has incorrect value. Limitations:\n" );
      printf( "nx >= nu, nx > 0, nu > 0, T > 0\n" );
      return 2;
    }

/*      Pack data into array and send it to other processes */
    iwork[ 0 ] = nx;
    iwork[ 1 ] = nu;
    iwork[ 2 ] = T;
    iwork[ 3 ] = total_size;
    iwork[ 4 ] = nominal_block_size;
    igebs2d_( &ictxt, "All", " ", &i_five, &i_one, iwork, &i_five );
  } else {

/*      Recieve and unpack data */
    igebr2d_( &ictxt, "All", " ", &i_five, &i_one, iwork, &i_five, &i_zero, &i_zero );
    nx = iwork[ 0 ];
    nu = iwork[ 1 ];
    T = iwork[ 2 ];
    total_size = iwork[ 3 ];
    nominal_block_size = iwork[ 4 ];
  }
  half_bw = 2*nx + nu - 1;
  band_matrix_rows = 4*half_bw + 1;


  if ( iam == 0 ) { fclose( fin ); }

  blacs_gridinit_( &ictxt_r, "R", &nproc, &i_one );
  blacs_gridinfo_( &ictxt_c, &nproc, &i_one, &myrow, &mycol );

/*  Compute precise length of local pieces and allocate array on
    each process for parts of distributed vectors */

  local_size = numroc_( &total_size, &nominal_block_size, &mycol, &i_zero, &nprocs );
  lwork = (local_size+half_bw)*(2*half_bw)+6*(2*half_bw)*3*half_bw+(local_size+6*half_bw);

  A = (double*) mkl_calloc(block_size*band_matrix_rows, sizeof( double ), 64);
  x = (double*) mkl_calloc(block_size, sizeof( double ), 64);
  b = (double*) mkl_calloc(block_size, sizeof( double ), 64);
  work = (double*) mkl_calloc(lwork, sizeof( double ), 64);

  int k;
  for (k = 0; k < block_size; ++k) {
    x[k] = rand();
    b[k] = rand();
  }
  for (k = 0; k < block_size*band_matrix_rows; ++k) {
    A[k] = rand();
  }

/*  Initialize descriptors for distributed arrays */
  descinit_( descA, &band_matrix_rows, &total_size, &band_matrix_rows, &nominal_block_size, &i_zero, &i_zero, &ictxt_c, &band_matrix_rows, &info );\
  descinit_( descb, &total_size, &i_one, &nominal_block_size, &i_one, &i_zero, &i_zero, &ictxt_r, &local_size, &info );
  descinit_( descx, &total_size, &i_one, &nominal_block_size, &i_one, &i_zero, &i_zero, &ictxt_r, &local_size, &info );

  blacs_barrier(ictxt_c, 'A');

  clock_t start = clock(), diff;
  pdgbsv_( &total_size, &half_bw, &half_bw, &i_one, A, &i_one, descA, &ipiv, B, &i_one, descb, work, &lwork, &info );

  blacs_barrier(ictxt_c, 'A');
  diff = clock() - start;

  int msec = diff * 1000 / CLOCKS_PER_SEC;
  if (iam == 0) {
    printf("Time taken %d seconds %d milliseconds \n", msec/1000, msec%1000);
  }

  mkl_free( work );
  mkl_free( A );
  mkl_free( x );
  mkl_free( b );

/*  Destroy process grid */
  blacs_gridexit_( &ictxt_r );
  blacs_gridexit_( &ictxt_c );
  blacs_exit_( &i_zero );

/*  Check if residual passed or failed the threshold */
  if ( ( iam == 0 ) && ( thresh >= zero ) && !( residual <= thresh ) ){
    printf( "FAILED. Residual = %05.16f\n", residual );
    return 1;
  } else {
    return 0;
  }

/*========================================================================
  ====== End of PBLAS Level 2 example ====================================
  ======================================================================*/
}