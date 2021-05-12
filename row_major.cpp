#include "../NiuTensor/source/tensor/XTensor.h"
#include "../NiuTensor/source/tensor/core/CHeader.h"
#include "../NiuTensor/source/tensor/function/FHeader.h"
#include <cstdlib>
#include <cstdio>
#include <immintrin.h>
using namespace std;
using namespace nts;

#define GEMM_SIMD_ALIGN_SIZE 32
#define DGEMM_MC 256
#define DGEMM_NC 256
#define DGEMM_KC 256
//#define DGEMM_KC 128
//#define DGEMM_MR 8
//#define DGEMM_NR 6
#define DGEMM_MR 8
#define DGEMM_NR 8

#define Min( i, j ) ( (i)<(j) ? (i): (j) )

//#define A( i, j )     A[ (j)*lda + (i) ]
//#define B( i, j )     B[ (j)*ldb + (i) ]
//#define C( i, j )     C[ (j)*ldc + (i) ]
//#define C_ref( i, j ) C_ref[ (j)*ldc_ref + (i) ]
//#define A( i, j )     A[ (i)*lda + (j) ]
//#define B( i, j )     B[ (i)*ldb + (j) ]
//#define C( i, j )     C[ (i)*ldc + (j) ]
//#define C_ref( i, j ) C_ref[ (i)*ldc_ref + (j) ]

struct aux_s {
    float *b_next;
    //float  *b_next_s;
    char   *flag;
    int    pc;
    int    m;
    int    n;
};
typedef struct aux_s aux_t;
////micro-panel a is stored in column major, lda=DGEMM_MR.
//#define a(i,j) a[ (j)*DGEMM_MR + (i) ]
////micro-panel b is stored in row major, ldb=DGEMM_NR.
//#define b(i,j) b[ (i)*DGEMM_NR + (j) ]
////result      c is stored in column major.
//#define c(i,j) c[ (j)*ldc + (i) ]

//micro-panel a is stored in column major, lda=DGEMM_MR.
#define a(i,j) a[ (j)*DGEMM_MR + (i) ]
//micro-panel b is stored in row major, ldb=DGEMM_NR.
#define b(i,j) b[ (i)*DGEMM_NR + (j) ]
//result      c is stored in column major.
#define c(i,j) c[ (i)*ldc + (j) ]

void bl_dgemm_int_8x8(
                        int    k,
                        float *a,
                        float *b,
                        float *c,
                        unsigned long long ldc,
                        aux_t* data
                      )
{
    int l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        __m256 rc;
        //__m256 rb = _mm256_broadcast_ss(b+l*DGEMM_NR);
        __m256 rb = _mm256_loadu_ps(b+l*DGEMM_NR);
        for(i=0;i<DGEMM_MR;i++)
        {
            __m256 ra = _mm256_broadcast_ss(a+l*DGEMM_MR+i);
            __m256 aux;
            //aux = _mm256_loadu_ps(c+i*DGEMM_NR);
            aux = _mm256_loadu_ps(c+i*ldc);
            //if(i==0)
                //aux = _mm256_loadu_ps(c+i*DGEMM_NR);
            //else
                //aux = _mm256_loadu_ps(c+(i-1)*DGEMM_NR);
            rc = _mm256_fmadd_ps(ra,rb,aux);
            //_mm256_storeu_ps(c+i*DGEMM_NR,rc);
            _mm256_storeu_ps(c+i*ldc,rc);
        }
    }
}

void bl_dgemm_ukr( int    k,
                   float *a,
                   float *b,
                   float *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        for ( j = 0; j < DGEMM_NR; ++j )
        { 
            for ( i = 0; i < DGEMM_MR; ++i )
            { 
                c( i, j ) += a( i, l ) * b( l, j );
            }
        }
    }

}

//#define BL_MICRO_KERNEL bl_dgemm_ukr
#define BL_MICRO_KERNEL bl_dgemm_int_8x8

static void (*bl_micro_kernel) (
        int    k,
        float *a,
        float *b,
        float *c,
        unsigned long long ldc,
        aux_t  *aux
        ) = {
        BL_MICRO_KERNEL
};
float *bl_malloc_aligned(
        int    m,
        int    n,
        int    size
        )
{
    float *ptr;
    int    err;

    err = posix_memalign( (void**)&ptr, (size_t)GEMM_SIMD_ALIGN_SIZE, size * m * n );

    if ( err ) {
        printf( "bl_malloc_aligned(): posix_memalign() failures" );
        exit( 1 );    
    }

    return ptr;
}
//#include "bl_dgemm_kernel.h"
//#include "bl_dgemm.h"

/*
 * @brief pack the block in col major format.
 *
 * @param m number of rows of the block
 * @param k number of columns of the block
 * @param XA The address of the first element of the matrix in the row corresponding to the first column of the block
 * e.g. 
 * a00 a01 a02 a03    the block is a22 a23
 * a10 a11 a12 a13                 a32 a33
 * a20 a21 a22 a23
 * a30 a31 a32 a33
 * then XA should be address of a20
 * @param ldXA leading dimension
 * @param offseta the col index of the first element of the block in the matrix
 * e.g. the matrix and block are shown above. then the offseta should be 2
 * @param packA store the packed block.
 * e.g. the matrix and block are shown above. then the packA should be
 * a22 a32 a23 a33
 */
inline void packA_mcxkc_d(
        int    m,
        int    k,
        float *XA,
        int    ldXA,
        int    offseta,
        float *packA
        )
{
    int    i, p;
    // @a_pntr store the address of each element in the first column of the block.
    float *a_pntr[ DGEMM_MR ];

    for ( i = 0; i < m; i ++ ) {
        //a_pntr[ i ] = XA + ( offseta + i );
        a_pntr[ i ] = XA + i*ldXA + offseta;
    }

    for ( i = m; i < DGEMM_MR; i ++ ) {
        a_pntr[ i ] = XA + ( offseta + 0 );
    }

    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < DGEMM_MR; i ++ ) {
            if(i>=m)
            {
                *packA=0;
                packA++;
                continue;
            }
            *packA = *a_pntr[ i ];
            packA ++;
            //a_pntr[ i ] = a_pntr[ i ] + ldXA;
            a_pntr[i]++;
        }
    }
}
//inline void packA_mcxkc_d(
        //int    m,
        //int    k,
        //float *XA,
        //int    ldXA,
        //int    offseta,
        //float *packA
        //)
//{
    //int    i, p;
    //// @a_pntr store the address of each element in the first column of the block.
    //float *a_pntr[ DGEMM_MR ];

    //for ( i = 0; i < m; i ++ ) {
        //a_pntr[ i ] = XA + ( offseta + i );
        ////a_pntr[ i ] = XA + i*ldXA + offseta;
    //}

    //for ( i = m; i < DGEMM_MR; i ++ ) {
        //a_pntr[ i ] = XA + ( offseta + 0 );
    //}

    //for ( p = 0; p < k; p ++ ) {
        //for ( i = 0; i < DGEMM_MR; i ++ ) {
            //*packA = *a_pntr[ i ];
            //packA ++;
            //a_pntr[ i ] = a_pntr[ i ] + ldXA;
            ////a_pntr[i]++;
        //}
    //}
//}


/*
 * @brief pack the block in row major format.
 *
 * @param n number of rows of the block
 * @param k number of columns of the block
 * @param XB The address of the first element of the matrix in the row corresponding to the first row of the block
 * e.g. 
 * b00 b01 b02 b03    the block is b22 b33
 * b10 b11 b12 b13                 b32 b33
 * b20 b21 b22 b23
 * b30 b31 b32 b33
 * then XB should be address of b20
 * @param ldXB leading dimension
 * @param offsetb the column index of the first element of the block in the matrix
 * e.g. the matrix and block are shown above. then the offsetb should be 2
 * @param packB store the packed block.
 * e.g. the matrix and block are shown above. then the packB should be
 * b22 b33 b32 b33
 */

inline void packB_kcxnc_d(
        int    n,
        int    k,
        float *XB,
        int    ldXB, // ldXB is the original n
        int    offsetb,
        float *packB
        )
{
    int    i, p; 
    // @b_pntr store the address of each element in the first row of the block.
    //float *b_pntr[ DGEMM_NR ];
    float *b_pntr[ DGEMM_KC ];

    for ( i = 0; i < k; i ++ ) {
        //b_pntr[ i ] = XB + ldXB * ( offsetb + j );
        b_pntr[ i ] = XB + ldXB * i + offsetb;
    }

    for ( i = k; i < DGEMM_KC; i ++ ) {
        b_pntr[ i ] = XB + ldXB * 0 +  offsetb;
    }

    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < DGEMM_NR; i ++ ) {
            if(i>=n)
            {
                *packB ++ = 0;
                continue;
            }
            *packB ++ = *b_pntr[ p ] ++;
        }
    }
}

/*
 * @param m MC
 * @param n NC
 * @param k KC
 * @param packA packed block A. size is MCxKC
 * @param packB packed block B. size is KCxNC
 * @param C the start address of corresponding block of the result matrix. notes that the matrix is stored in column major.
 */
void bl_macro_kernel(
        int    m,
        int    n,
        int    k,
        float *packA,
        float *packB,
        float *C,
        int    ldc
        )
{
    int bl_ic_nt;
    int    i, ii, j;
    aux_t  aux;
    char *str;

    aux.b_next = packB;

    for ( j = 0; j < n; j += DGEMM_NR ) {                        // 2-th loop around micro-kernel
        aux.n  = Min( n - j, DGEMM_NR );
        for ( i = 0; i < m; i += DGEMM_MR ) {                    // 1-th loop around micro-kernel
            aux.m = Min( m - i, DGEMM_MR );
            // what does this piece of code mean ?
            if ( i + DGEMM_MR >= m ) {
                aux.b_next += DGEMM_NR * k;
            }

            ( *bl_micro_kernel ) (
                    k,
                    &packA[ i * k ],
                    &packB[ j * k ],
                    &C[ i * ldc + j ],
                    (unsigned long long) ldc,
                    &aux
                    );
        }                                                        // 1-th loop around micro-kernel
    }                                                            // 2-th loop around micro-kernel
}

// C must be aligned
void bl_dgemm(
        int    m,
        int    n,
        int    k,
        float *XA,
        int    lda,
        float *XB,
        int    ldb,
        float *C,        // must be aligned
        int    ldc        // ldc must also be aligned
        //float *XA_TRANS
        )
{
    int    i, j, p, bl_ic_nt;
    int    ic, ib, jc, jb, pc, pb;
    int    ir, jr;
    float *packA, *packB, *packA_ref;
    char   *str;

    // Early return if possible
    if ( m == 0 || n == 0 || k == 0 ) {
        printf( "bl_dgemm(): early return\n" );
        return;
    }

    // sequential is the default situation
    bl_ic_nt = 1;
    // check the environment variable
    //str = getenv( "BLISLAB_IC_NT" );
    //if ( str != NULL ) {
        //bl_ic_nt = (int)strtol( str, NULL, 10 );
    //}

    // Allocate packing buffers
    // packA is KCxMC by default. so it only stores one block??? strange...
    packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ) * bl_ic_nt, sizeof(float) );
    packA_ref  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ) * bl_ic_nt, sizeof(float) );
    // packB is KCxNC. so it only stores one block??? strange...
    packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC + 1 )           , sizeof(float) );

    for ( jc = 0; jc < n; jc += DGEMM_NC ) {                                       // 5-th loop around micro-kernel
        jb = Min( n - jc, DGEMM_NC );
        for ( pc = 0; pc < k; pc += DGEMM_KC ) {                                   // 4-th loop around micro-kernel
        // so for each pc, we will update packA and packB.
            // @pb row number of a block
            pb = Min( k - pc, DGEMM_KC );

            // so for B, the pack order is: K->N
            // e.g. the B is b00 b01 b02 b03 and KC is 2, NC is 3, NR is 2.
            //               b10 b11 b12 b13
            //               b20 b21 b22 b23
            //               b30 b31 b32 b33
            // 
            //                         1              2               3      4
            // then the pack order is: b00 b01 b02 -> b20 b21 b22 ->  b03 -> b23 
            //                         b10 b11 b12    b30 b31 b32     b13    b33
            //
            // for each pc, the packB will be
            // b00 b01 b02 b10 b11 b12
            // and
            // b20 b21 b22 b30 b31 b32
            // and
            // not sure what will fill the block, but may be it is not important.
            // b03 b13 x x x x(?)

            // I don't know why we need an NR while we have NC.
            for ( j = 0; j < jb; j += DGEMM_NR ) {
                // so I want to know the order in which B is packed.
                packB_kcxnc_d(
                        Min( jb - j, DGEMM_NR ),
                        pb,
                        &XB[ pc*ldb ],
                        ldb, // should be ldXB instead
                        jc + j,
                        &packB[ j * pb ]
                        );
            }


            // so for A, the pack order is: K->M
            //for ( ic = 0; ic < m; ic += DGEMM_MC ) {                               // 3-rd loop around micro-kernel
            for ( ic = 0; ic < m; ic += DGEMM_MC ) {                               // 3-rd loop around micro-kernel
            // so for each ic, we execute a kernel computation.

                ib = Min( m - ic, DGEMM_MC );

                for ( i = 0; i < ib; i += DGEMM_MR ) {
                    packA_mcxkc_d(
                            Min( ib - i, DGEMM_MR ),
                            pb,
                            &XA[ (ic+i)*lda ],
                            lda,
                            pc,
                            &packA[ 0 * DGEMM_MC * pb + i * pb ]
                            );
                }

                //ib = min( m - ic, DGEMM_MC );
                //jb = min( n - jc, DGEMM_NC );
                //pb = min( k - pc, DGEMM_KC );
                //packA  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_MC + 1 ) * bl_ic_nt, sizeof(double) );
                //packB  = bl_malloc_aligned( DGEMM_KC, ( DGEMM_NC + 1 )           , sizeof(double) );
                //for ( jc = 0; jc < n; jc += DGEMM_NC ) {                                       // 5-th loop around micro-kernel
                //for ( ic = 0; ic < m; ic += DGEMM_MC ) {                               // 3-rd loop around micro-kernel
                bl_macro_kernel(
                        ib,
                        jb,
                        pb,
                        // wtf??? the expression 0 * DGEMM_MC * pb always returns 0. so why we need it???
                        // so when I look at the corresponding code in step4, a tid replace the 0. so may be the expression is about omp parallelize?
                        packA  + 0 * DGEMM_MC * pb,
                        //packA_ref  + 0 * DGEMM_MC * pb,
                        packB,
                        &C[ ic * ldc + jc ], 
                        ldc
                        );
            }                                                                     // End 3.rd loop around micro-kernel
        }                                                                         // End 4.th loop around micro-kernel
    }                                                                             // End 5.th loop around micro-kernel

    free( packA );
    free( packB );
}

void calDiff(XTensor a,XTensor b,XTensor scale)
{
    float diff=0;
    long long cnt=0;
    FILE* f1,*f2;
    f1 = fopen("blis","w");
    f2 = fopen("ref","w");
    for(int i=0;i<20;++i)
    {
        printf("=");
    }
    printf("\n");
    for(int i=0;i<a.dimSize[0];++i)
    {
	    for(int j=0;j<a.dimSize[1];++j)
	    {
            //printf("\n%d,%d\n",i,j);
            float av = a.Get2D(i,j);
            float bv = b.Get2D(i,j);
            fprintf(f1,"%.3f\n",av);
            fprintf(f2,"%.3f\n",bv);
            float s = scale.Get2D(i,j);
            if((av>bv?av-bv:bv-av)>0.1)
                cnt++;
            diff += ((av>bv?av-bv:bv-av)/s);
	    }
    }
    printf("diff per:%f\n",diff/(a.unitNum));
    printf("total diff:%f\n",diff);
    printf("err num:%lld\n",cnt);
    fclose(f1);
    fclose(f2);
}
int main()
{
	XTensor a,b,c_ref,c,a_trans;
    int m,k,n;
    m=509;
    k=513;
    n=513;
    InitTensor2D(&a,m,k,nts::X_FLOAT);
    InitTensor2D(&a_trans,m,k,nts::X_FLOAT);
	InitTensor2D(&b, k, n, nts::X_FLOAT);
    InitTensor2D(&c_ref,m,n,X_FLOAT);
    InitTensor2D(&c,m,n,X_FLOAT);
    a.SetDataRand(-1,1);
    b.SetDataRand(-1,1);
    c.SetZeroAll();
    c_ref=MatrixMul(a,b);
    //a_trans=Transpose(a,0,1);
    //b=Transpose(b,0,1);
    bl_dgemm(
            m,
            n,
            k,
            (float*)a.data,
            k,
            (float*)b.data,
            n,
            (float*)c.data,
            n
            //(float*)a_trans.data
            );
    //printf("\n%d,%d\n",c.dimSize[0],c.dimSize[1]);
    //c=Transpose(c,0,1);
    //printf("\n%d,%d\n",c.dimSize[0],c.dimSize[1]);
    //exit(1);
    XTensor s;
    InitTensor2D(&s,m,n,X_FLOAT);
    s.SetDataFixed(1);
    calDiff(c,c_ref,s);
}
