/* covariance.c: this file is part of PolyBench/C */

#include <stdio.h>
//#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include "covariance.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)n;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = ((DATA_TYPE) i*j) / M;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(cov,M,M,m,m))

{
  return;
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("cov");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, cov[i][j]);
    }
  POLYBENCH_DUMP_END("cov");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_covariance(int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		       DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, k;

//#pragma scop
  
#pragma clang loop(loop7,loop8,loop9) tile sizes(64,256,4) floor_ids(loop372,loop366,loop328) tile_ids(loop373,loop367,loop329)

#pragma clang loop id(loop1)
for (j = 0; j < _PB_M; j++) {
      mean[j] = SCALAR_VAL(0.0);
      
#pragma clang loop id(loop2)
for (i = 0; i < _PB_N; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
  }

  
#pragma clang loop id(loop3)
for (i = 0; i < _PB_N; i++)
    
#pragma clang loop id(loop4)
for (j = 0; j < _PB_M; j++)
      data[i][j] -= mean[j];

  
#pragma clang loop id(loop5)
for (i = 0; i < _PB_M; i++) {
    
#pragma clang loop id(loop6)
for (j = i; j < _PB_M; j++) {
        cov[i][j] = SCALAR_VAL(0.0);
    }
  }

  
#pragma clang loop id(loop7)
for (i = 0; i < _PB_M; i++) {
    
#pragma clang loop id(loop8)
for (j = i; j < _PB_M; j++) {
        
#pragma clang loop id(loop9)
for (k = 0; k < _PB_N; k++)
	        cov[i][j] += data[k][i] * data[k][j];
    }
  }

  
#pragma clang loop id(loop10)
for (i = 0; i < _PB_M; i++) {
    
#pragma clang loop id(loop11)
for (j = i; j < _PB_M; j++) {
        cov[i][j] /= (float_n - SCALAR_VAL(1.0));
        cov[j][i] = cov[i][j];
    }
  }
//#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(cov,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);


  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_covariance (m, n, float_n,
		     POLYBENCH_ARRAY(data),
		     POLYBENCH_ARRAY(cov),
		     POLYBENCH_ARRAY(mean));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(cov)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(cov);
  POLYBENCH_FREE_ARRAY(mean);

  return 0;
}