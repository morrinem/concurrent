//
// CSU33014 Lab 1
//

// Please examine version each of the following routines with names
// starting lab1. Where the routine can be vectorized, please
// complete the corresponding vectorized routine using SSE vector
// intrinsics.

// Note the restrict qualifier in C indicates that "only the pointer
// itself or a value directly derived from it (such as pointer + 1)
// will be used to access the object to which it points".


#include <immintrin.h>
#include <stdio.h>

#include "lab1-code.h"

void printVector(__m128 a) {
	float temp[4];
	_mm_storeu_ps(temp, a);
	printf("[%f],[%f],[%f],[%f]\n", temp[0], temp[1], temp[2], temp[3]);
}

void printVectorHex(__m128 a) {
	unsigned int temp[4];
	_mm_storeu_ps((float *)temp, a);
	printf("[%#010x],[%#010x],[%#010x],[%#010x]\n", temp[0], temp[1], temp[2], temp[3]);
}

/****************  routine 0 *******************/

// Here is an example routine that should be vectorized
void lab1_routine0(float * restrict a, float * restrict b,
		    float * restrict c) {
  for (int i = 0; i < 1024; i++ ) {
    a[i] = b[i] * c[i];
  }
}

// here is a vectorized solution for the example above
void lab1_vectorized0(float * restrict a, float * restrict b,
		    float * restrict c) {
  __m128 a4, b4, c4;
  
  for (int i = 0; i < 1024; i = i+4 ) {
    b4 = _mm_loadu_ps(&b[i]);
    c4 = _mm_loadu_ps(&c[i]);
    a4 = _mm_mul_ps(b4, c4);
    _mm_storeu_ps(&a[i], a4);
  }
}

/***************** routine 1 *********************/

// in the following, size can have any positive value
float lab1_routine1(float * restrict a, float * restrict b,
		     int size) {
  float sum = 0.0;
  
  for ( int i = 0; i < size; i++ ) {
    sum = sum + a[i] * b[i];
  }
  return sum;
}

// insert vectorized code for routine1 here
float lab1_vectorized1(float * restrict a, float * restrict b,
		     int size) {
  __m128 a4, b4, result4;
  result4 = _mm_setzero_ps();
  int remainder = size%4;
  int i;
  for (i = 0; i < size-remainder; i+=4)
  {
	a4 = _mm_loadu_ps(&a[i]);
	b4 = _mm_loadu_ps(&b[i]);
	result4 = _mm_add_ps(result4, _mm_mul_ps(a4, b4));
  }
  result4 = _mm_hadd_ps(result4, result4);
  result4 = _mm_hadd_ps(result4, result4);
  float temp[4];
  _mm_storeu_ps(temp, result4);
  float result = temp[0];
  for ( ; i < size; i++)
  {
	result += a[i]*b[i];
  }
  return result;
}

/******************* routine 2 ***********************/

// in the following, size can have any positive value
void lab1_routine2(float * restrict a, float * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    a[i] = 1 - (1.0/(b[i]+1.0));
  }
}

// in the following, size can have any positive value
void lab1_vectorized2(float * restrict a, float * restrict b, int size) {
  __m128 a4, b4;
  __m128 all1 = _mm_set1_ps(1.0);
  int remainder = size%4;
  int i;
  for (i=0; i < size-remainder; i+=4)
  {
	b4 = _mm_loadu_ps(&b[i]);
	a4 = _mm_add_ps(b4, all1);
	a4 = _mm_rcp_ps(a4);
	a4 = _mm_sub_ps(all1, a4);
	_mm_storeu_ps(&a[i], a4);
  }
  for ( ; i < size; i++) a[i] = 1-(1.0/(b[i]+1.0));

}

/******************** routine 3 ************************/

// in the following, size can have any positive value
void lab1_routine3(float * restrict a, float * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    if ( a[i] < 0.0 ) {
      a[i] = b[i];
    }
  }
}

// in the following, size can have any positive value
void lab1_vectorized3(float * restrict a, float * restrict b, int size) {
  __m128 a4, b4;
  __m128 allzero = _mm_setzero_ps();
  int remainder = size%4;
  int i;
  for (i=0; i < size-remainder; i+=4){
	a4 = _mm_loadu_ps(&a[i]);
	b4 = _mm_loadu_ps(&b[i]);
	__m128 mask = _mm_cmplt_ps(a4, allzero);
	__m128 notmask = _mm_cmpge_ps(a4, allzero);
	__m128 resulta = _mm_and_ps(a4, notmask);
	__m128 resultb = _mm_and_ps(b4, mask);
	__m128 result = _mm_or_ps(resulta, resultb);
	_mm_storeu_ps(&a[i], result);
  }
  for ( ; i < size; i++ ) {
    if ( a[i] < 0.0 ) {
      a[i] = b[i];
    }
  }
}

/********************* routine 4 ***********************/

// hint: one way to vectorize the following code might use
// vector shuffle operations
void lab1_routine4(float * restrict a, float * restrict b,
		       float * restrict c) {
  for ( int i = 0; i < 2048; i = i+2  ) {
    a[i] = b[i]*c[i] - b[i+1]*c[i+1];
    a[i+1] = b[i]*c[i+1] + b[i+1]*c[i];
  }
}

void lab1_vectorized4(float * restrict a, float * restrict b,
		       float * restrict  c) {
  __m128 a4, b4, c4;
  for (int i = 0; i < 2048; i+=4)
  {
	b4 = _mm_loadu_ps(&b[i]);
	c4 = _mm_loadu_ps(&c[i]);
	__m128 a0r = _mm_mul_ps(b4, c4);
	a0r = _mm_hsub_ps(a0r, a0r);
	
	__m128 cshift = _mm_shuffle_ps(c4, c4, _MM_SHUFFLE(2, 3, 0, 1));
	__m128 a1r = _mm_mul_ps(b4, cshift);
	a1r = _mm_hadd_ps(a1r, a1r);
	
	float a0f[4], a1f[4];
	_mm_storeu_ps(a0f, a0r);
	_mm_storeu_ps(a1f, a1r);
	a4 = _mm_set_ps(a1f[1], a0f[1], a1f[0], a0f[0]);
	_mm_storeu_ps(&a[i], a4);
  }
}

/********************* routine 5 ***********************/

// in the following, size can have any positive value
void lab1_routine5(unsigned char * restrict a,
		    unsigned char * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    a[i] = b[i];
  }
}

void lab1_vectorized5(unsigned char * restrict a,
		       unsigned char * restrict b, int size) {
  int remainder = size%16;
  int i;
  for (i=0; i < size-remainder; i+= 16)
  {
	__m128 tempchar = _mm_loadu_ps((float*)&b[i]);
	_mm_storeu_ps((float*)&a[i], tempchar);
  }
  for ( ;i < size; i++ ) {
    a[i] = b[i];
  }
}

/********************* routine 6 ***********************/

void lab1_routine6(float * restrict a, float * restrict b,
		       float * restrict c) {
  a[0] = 0.0;
  for ( int i = 1; i < 1023; i++ ) {
    float sum = 0.0;
    for ( int j = 0; j < 3; j++ ) {
      sum = sum +  b[i+j-1] * c[j];
    }
    a[i] = sum;
  }
  a[1023] = 0.0;
}

void lab1_vectorized6(float * restrict a, float * restrict b,
		       float * restrict c) {
  __m128 a4, b4, c4;
  c4 = _mm_set_ps(0.0, c[2], c[1], c[0]);
  a[0] = 0.0;
  for (int i = 1; i < 1023; i++)
  {
	b4 = _mm_loadu_ps(&b[i-1]);
	a4 = _mm_mul_ps(b4, c4);
	a4 = _mm_hadd_ps(a4, a4);
	a4 = _mm_hadd_ps(a4, a4);
	float temp[4];
	_mm_storeu_ps(temp, a4);
	a[i]=temp[0];
  }
  a[1023] = 0.0;
}



