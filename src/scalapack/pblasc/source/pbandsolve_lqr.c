//
// Created by Forrest Laine on 8/20/18.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl_scalapack.h>
#include <time.h>
#include "mkl.h"

/* Definition of MIN and MAX functions */
#define MAX(a,b)((a)<(b)?(b):(a))
#define MIN(a,b)((a)>(b)?(b):(a))

#define MAXINT 100
#define MININT -100

/* Definition of matrix descriptor */
typedef MKL_INT MDESC[ 9 ];

/* Parameters */
const double zero = 0.0E+0, one = 1.0E+0, two = 2.0E+0;
const MKL_INT i_zero = 0, i_one = 1, i_five = 5, i_negone = -1;
const char trans = 'N';


/*==== MAIN FUNCTION =================================================*/
int main(int argc, char *argv[]) {

/*  ==== Declarations =================================================== */

/*  File variables */
  FILE *fin;

/*  Matrix descriptors */
  MDESC descA, descb, descx;
  MKL_INT descA_b[7];
  MKL_INT descb_b[7];

/*  Local scalars */
  MKL_INT iam, nprocs, ictxt_r, ictxt_c, myrow, mycol, nrows, ncols;
  MKL_INT nominal_block_size, lld;
  MKL_INT T, nx, nu, total_size, local_size;
  MKL_INT half_bw, band_matrix_rows, lwork, info, ipiv;

  int nx_int, nu_int, T_int;

/*  Local arrays */
  double *A, *x, *b, *work;
  MKL_INT iwork[5];

  /*  ==== Executable statements ========================================== */

/*  Get information about how many processes are used for program execution
    and number of current process */
  blacs_pinfo_( &iam, &nprocs );

  srand(iam);
/*  Init temporary 1D process grid */
  blacs_get_( &i_negone, &i_zero, &ictxt_c );
  blacs_gridinit_( &ictxt_c, "R", &i_one, &nprocs  );

/*  Open input file */
  if ( iam == 0 ) {
    fin = fopen( "../in/lqr.in", "r" );
    if ( fin == NULL ) {
      printf( "Error while open input file." );
      return 2;
    }
  }

/*  Read data and send it to all processes */
  if ( iam == 0 ) {
/*      Read parameters */
    fscanf( fin, "%d nx, dimension of state vectors, must be > 0 ", &nx_int );
    fscanf( fin, "%d nu, dimension of control vectors, must be > 0 ", &nu_int );
    fscanf( fin, "%d T, trajectory length, must be > 0 ", &T_int );

    T = (MKL_INT) T_int;
    nx = (MKL_INT) nx_int;
    nu = (MKL_INT) nu_int;
    total_size = (2 * nx + nu) * (T-1) + 2 * nx;
    nominal_block_size = (MKL_INT) ceil(total_size / (1.0*nprocs));

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
    igebs2d_( &ictxt_c, "All", " ", &i_one, &i_five, iwork, &i_one );
  } else {

/*      Recieve and unpack data */
    igebr2d_( &ictxt_c, "All", " ", &i_one, &i_five, iwork, &i_one, &i_zero, &i_zero );
    nx = iwork[ 0 ];
    nu = iwork[ 1 ];
    T = iwork[ 2 ];
    total_size = iwork[ 3 ];
    nominal_block_size = iwork[ 4 ];
  }

  half_bw = (MKL_INT) (2*nx + nu - 1);
  band_matrix_rows = (MKL_INT) (2*half_bw + 1);

  if ( iam == 0 ) { fclose( fin ); }
  
  blacs_gridinfo_( &ictxt_c, &nrows, &ncols, &myrow, &mycol );
/*  Compute precise length of local pieces and allocate array on
    each process for parts of distributed vectors */

  local_size = numroc_( &total_size, &nominal_block_size, &mycol, &i_zero, &nprocs );
  if(iam==0) printf("Total size: %d\n", total_size);
  printf("Local size: %d\n", local_size);
  lwork = (MKL_INT) 10*((local_size+half_bw)*(2*half_bw)+6*(2*half_bw)*3*half_bw+(local_size+6*half_bw));
  

  A = (double*) mkl_calloc(local_size*band_matrix_rows, sizeof( double ), 64);
  b = (double*) mkl_calloc(local_size, sizeof( double ), 64);
  work = (double*) mkl_calloc(lwork, sizeof( double ), 64);

  int k;
  for (k = 0; k < local_size; ++k) {
    b[k] = MININT + rand() / (RAND_MAX / (MAXINT - MININT + 1.0) + 1.0);
  }
  int z, idx;
  idx = 0;

  for (k = 0; k < local_size; ++k) {
  	for (z = 0; z < band_matrix_rows; ++z) {
		//if( z > 2*half_bw) {
			A[idx] = MININT + rand() / (RAND_MAX / (MAXINT - MININT + 1.0) + 1.0);
		//} else {
		//	A[idx] = 0.;
		//}
		idx += 1;
	}
  }
  blacs_barrier_(&ictxt_c, "A");
/*  Initialize descriptors for distributed arrays */
  descA_b[0] = (MKL_INT) 501;
  descA_b[1] = ictxt_c;
  descA_b[2] = total_size;
  descA_b[3] = local_size;
  descA_b[4] = i_zero;
  descA_b[5] = band_matrix_rows;

  descb_b[0] = (MKL_INT) 502;
  descb_b[1] = ictxt_c;
  descb_b[2] = total_size;
  descb_b[3] = local_size;
  descb_b[4] = i_zero;
  descb_b[5] = local_size;
 
  //descinit_( descA, &band_matrix_rows, &total_size, &band_matrix_rows, &nominal_block_size, &i_zero, &i_zero, &ictxt_c, &band_matrix_rows, &info );
  //descinit_( descb, &total_size, &i_one, &nominal_block_size, &i_one, &i_zero, &i_zero, &ictxt_c, &local_size, &info );
  blacs_barrier_(&ictxt_c, "A");
  clock_t start = clock(), diff;


  //pdgbsv_( &total_size, &half_bw, &half_bw, &i_one, A, &i_one, descA_b, &ipiv, b, &i_one, descb_b, work, &lwork, &info );
  pddbsv_( &total_size, &half_bw, &half_bw, &i_one, A, &i_one, descA_b, b, &i_one, descb_b, work, &lwork, &info );
  //blacs_barrier_(&ictxt_c, "A");
  diff = clock() - start;

  double msec = (double) diff / CLOCKS_PER_SEC;
  if(iam==0)  printf("Time taken %f seconds\n", msec);
 
  mkl_free( work );
  mkl_free( A );
  mkl_free( b );
/*  Destroy process grid */
  blacs_gridexit_( &ictxt_c );
  blacs_exit_( &i_zero );

/*  Check if residual passed or failed the threshold */
  if ( iam == 0 && info != 0 ){
    printf( "FAILED. \n" );
    return 1;
  } else {
    return 0;
  }

/*========================================================================
  ====== End of PBLAS Level 2 example ====================================
  ======================================================================*/
}
