/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

// Block sizes
#define mc 256
#define kc 128

#define min(i, j) ( (i) < (j) ? (i) : (j) )

/* Routine for computing C = A * B + C */

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc);

void MY_MMult( int m, int n, int k, double *a, int lda, 
		double *b, int ldb,
		double *c, int ldc )
{
	int i, j, p, pb, ib;

	// This time, we compute a mc x n block of C by a call to the InnerKernel 
	for (p = 0; p < k; p+= kc){
		pb = min(k-p, kc);
		for ( i = 0; i < m; i += mc){
                    ib = min(m-i , mc);
                    InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc); 
		}
	}

}


void InnerKernel( int m, int n, int k, double *a, int lda, 
		double *b, int ldb,
		double *c, int ldc )
{
	int i, j;

	for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
		for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C , unrolled by 4*/
			/* Update the C( i,j ) C( i,j+1) C( i,j+2) C( i,j+3) with the inner product of the ith row of A
			   and the jth column of B (four inner products) */

			AddDot4x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc);
		}
	}
}


#include <mmintrin.h>
#include <xmmintrin.h> // SSE, 128, 8 32bits, 16 64bits
#include <pmmintrin.h> // SSE2
#include <emmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE 4.1
#include <nmmintrin.h> // SSE 4.2
#include <immintrin.h> // AVX, 256
// AVX512, 512

typedef union{
    // __mXXX(T)
    // XXX is the number of bits of the vector (128 for SSE, 256 for AVX)
    // T is omitted for float, i fot integers and d for double
    __m128d v;     // 16 bytes, 128 bits
    double d[2];  // 2 double 
    // float f[4];   // 4 float 
    // int32_t i[4];  // 4 int32
} v2df_t;

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc){

	/* So, this routine computes a 4x4 block of matrix A
	   C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
	   C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
	   C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
	   C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  
	   Notice that this routine is called with c = C( i, j ) in the
	   previous routine, so these are actually the elements 
	   C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
	   C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
	   C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
	   C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 

	   in the original matrix C
	   And now we user vector registers and instructions
	 */ 

	int p;
	v2df_t 
            c_00_c_10_vreg, c_01_c_11_vreg, c_02_c_12_vreg, c_03_c_13_vreg,
            c_20_c_30_vreg, c_21_c_31_vreg, c_22_c_32_vreg, c_23_c_33_vreg,
            a_0p_a_1p_vreg,
            a_2p_a_3p_vreg,
            b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;


	/* Point to the current elements in the four columns of B */
	double *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr; 


        // B(p, j)
	b_p0_pntr = &B( 0, 0 );
	b_p1_pntr = &B( 0, 1 );
	b_p2_pntr = &B( 0, 2 );
	b_p3_pntr = &B( 0, 3 );

        // _mm(XXX)_NAME_PT
        // XXX is the number of bits of the SIMD registers; it is omitted for 128 bits registers
        // P indicates whether the functions operates on a packed data vector (p) or on a scalar only (s)
        // T indicates the type of the floating point numbers : s for single precision, d for double precision
        c_00_c_10_vreg.v = _mm_setzero_pd();
	c_01_c_11_vreg.v = _mm_setzero_pd();
	c_02_c_12_vreg.v = _mm_setzero_pd(); 
	c_03_c_13_vreg.v = _mm_setzero_pd(); 
	c_20_c_30_vreg.v = _mm_setzero_pd();   
	c_21_c_31_vreg.v = _mm_setzero_pd();  
	c_22_c_32_vreg.v = _mm_setzero_pd();   
	c_23_c_33_vreg.v = _mm_setzero_pd(); 

	for ( p=0; p<k; p++ ){
		//a_0p_reg = A( 0, p );
		//a_1p_reg = A( 1, p );
		//a_2p_reg = A( 2, p );
		//a_3p_reg = A( 3, p );
                a_0p_a_1p_vreg.v = _mm_load_pd( (double*) & A(0, p));
                a_2p_a_3p_vreg.v = _mm_load_pd( (double*) & A(2, p));

		//b_p0_reg = *b_p0_pntr++;
		//b_p1_reg = *b_p1_pntr++;
		//b_p2_reg = *b_p2_pntr++;
		//b_p3_reg = *b_p3_pntr++;	
                b_p0_vreg.v = _mm_loaddup_pd( (double*) b_p0_pntr++); // load and duplicate
                b_p1_vreg.v = _mm_loaddup_pd( (double*) b_p1_pntr++); // load and duplicate
                b_p2_vreg.v = _mm_loaddup_pd( (double*) b_p2_pntr++); // load and duplicate
                b_p3_vreg.v = _mm_loaddup_pd( (double*) b_p3_pntr++); // load and duplicate


		/* First row and second rows */
		//c_00_reg += a_0p_reg * b_p0_reg;
		//c_10_reg += a_1p_reg * b_p0_reg;
                c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;

		//c_01_reg += a_0p_reg * b_p1_reg;
		//c_11_reg += a_1p_reg * b_p1_reg;
		c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;

		//c_02_reg += a_0p_reg * b_p2_reg;
		//c_12_reg += a_1p_reg * b_p2_reg;
		c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;

		//c_03_reg += a_0p_reg * b_p3_reg;
		//c_13_reg += a_1p_reg * b_p3_reg;
		c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

		/* Third and fourth rows */
		//c_20_reg += a_2p_reg * b_p0_reg;
		//c_30_reg += a_3p_reg * b_p0_reg;
		c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;

		//c_21_reg += a_2p_reg * b_p1_reg;
		//c_31_reg += a_3p_reg * b_p1_reg;
		c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;

		//c_22_reg += a_2p_reg * b_p2_reg;
		//c_32_reg += a_3p_reg * b_p2_reg;
		c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;

		// c_23_reg += a_2p_reg * b_p3_reg;
		// c_33_reg += a_3p_reg * b_p3_reg;
		c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;


	}


	C( 0, 0 ) += c_00_c_10_vreg.d[0];  C( 0, 1 ) += c_01_c_11_vreg.d[0];  
	C( 0, 2 ) += c_02_c_12_vreg.d[0];  C( 0, 3 ) += c_03_c_13_vreg.d[0]; 

	C( 1, 0 ) += c_00_c_10_vreg.d[1];  C( 1, 1 ) += c_01_c_11_vreg.d[1];  
	C( 1, 2 ) += c_02_c_12_vreg.d[1];  C( 1, 3 ) += c_03_c_13_vreg.d[1]; 

	C( 2, 0 ) += c_20_c_30_vreg.d[0];  C( 2, 1 ) += c_21_c_31_vreg.d[0];  
	C( 2, 2 ) += c_22_c_32_vreg.d[0];  C( 2, 3 ) += c_23_c_33_vreg.d[0]; 

	C( 3, 0 ) += c_20_c_30_vreg.d[1];  C( 3, 1 ) += c_21_c_31_vreg.d[1];  
	C( 3, 2 ) += c_22_c_32_vreg.d[1];  C( 3, 3 ) += c_23_c_33_vreg.d[1]; 
}
