/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot1x4( int, double *, int, double *, int, double *, int);

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  // unroll N
  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    for ( i=0; i<m; i+=1 ){        /* Loop over the rows of C */
      /* Update the C( i,j ), C(i,j+1), C(i, j+2) and C(i, j+3) in one
        routine (four inner products of the ith row of A
	 and the jth column of B */

      AddDot1x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc);
    }
  }
}


void AddDot1x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){
  /* So, this routine computes four elements of C: 

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i, j ), C( i, j+1 ), C( i, j+2 ), C( i, j+3 ) 
	  
     in the original matrix C

     In this version, we use pointer to track where in four columns of B we are.
  */ 

    int p;
    /* hold contributions to C(0,0), C(0, 1), C(0,2), C(0,3) A(0, p) */
    register double c_00_reg, c_01_reg, c_02_reg, c_03_reg, a_0p_reg;
    double *b_p0_ptr, *b_p1_ptr, *b_p2_ptr, *b_p3_ptr;

    b_p0_ptr = &B(0, 0);
    b_p1_ptr = &B(0, 1);
    b_p2_ptr = &B(0, 2);
    b_p3_ptr = &B(0, 3);

    c_00_reg = 0.0;
    c_01_reg = 0.0;
    c_02_reg = 0.0;
    c_03_reg = 0.0;

    // unroll K
    for (p = 0; p < k; p += 4 ){
       a_0p_reg = A(0, p);
       // Note: col-major, B(p, 0)
       c_00_reg += a_0p_reg * *b_p0_ptr;
       c_01_reg += a_0p_reg * *b_p1_ptr;
       c_02_reg += a_0p_reg * *b_p2_ptr;
       c_03_reg += a_0p_reg * *b_p3_ptr;

       a_0p_reg = A(0, p+1);
       c_00_reg += a_0p_reg * *(b_p0_ptr+1);
       c_01_reg += a_0p_reg * *(b_p1_ptr+1);
       c_02_reg += a_0p_reg * *(b_p2_ptr+1);
       c_03_reg += a_0p_reg * *(b_p3_ptr+1);

       a_0p_reg = A(0, p+2);
       c_00_reg += a_0p_reg * *(b_p0_ptr+2);
       c_01_reg += a_0p_reg * *(b_p1_ptr+2);
       c_02_reg += a_0p_reg * *(b_p2_ptr+2);
       c_03_reg += a_0p_reg * *(b_p3_ptr+2);

       a_0p_reg = A(0, p+3);
       c_00_reg += a_0p_reg * *(b_p0_ptr+3);
       c_01_reg += a_0p_reg * *(b_p1_ptr+3);
       c_02_reg += a_0p_reg * *(b_p2_ptr+3);
       c_03_reg += a_0p_reg * *(b_p3_ptr+3);


       b_p0_ptr += 4;
       b_p1_ptr += 4;
       b_p2_ptr += 4;
       b_p3_ptr += 4;
    }

    C(0, 0) += c_00_reg;
    C(0, 1) += c_01_reg;
    C(0, 2) += c_02_reg;
    C(0, 3) += c_03_reg;
}
