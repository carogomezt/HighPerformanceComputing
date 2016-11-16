#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

void Mult_RowMat(float *M_a, float *M_b, float *R_c, int NRA, int NCA, int NCB);

int main (int argc, char *argv[])
{
float *a,           /* matrix A to be multiplied */
    *b,           /* matrix B to be multiplied */
    *c;           /* result matrix C */

  int NRA =  5;
  int NCA =  5;
  int NCB =  5;

  a = (float*)malloc(NRA * NCA * sizeof(float));
  b = (float*)malloc(NCA * NCB * sizeof(float));
  c = (float*)malloc(NRA * NCB * sizeof(float));

  for (int i = 0; i < NRA; i++) {
    for (int j = 0; j < NCA; j++){
      a[i * NCA + j] = 1;
    }
  }

  for (int i = 0; i < NCA; i++) {
    for (int j = 0; j < NCB; j++){
      b[i * NCB + j] = 1;
    }
  }

  Mult_RowMat(a, b, c, NRA, NCA, NCB);

  /* Print results */
  printf("******************************************************\n");
  printf("Result Matrix:\n");
  
  for (int i = 0; i < NRA; i++) {
    for (int j = 0; j < NCB; j++){
      cout<< c[i * NCB + j]<<" ";
    }
    cout<<endl;
  }

  printf("\n******************************************************\n");
  printf ("Done.\n");

  free(a);
  free(b);
  free(c);
}