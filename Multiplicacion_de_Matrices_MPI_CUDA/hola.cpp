#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

#define NRA 5                 
#define NCA 5
#define NCB 5

void Mult_RowMat(float *R_a, float *M_b, float *R_c, int H, int W);

int main (int argc, char *argv[])
{
float *a,           /* matrix A to be multiplied */
    *b,           /* matrix B to be multiplied */
    *c;           /* result matrix C */

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

  for (int i = 0; i < NRA; ++i) {
    Mult_RowMat(&a[i* NCA], b, &c[i * NCB], NCA, NCB);
  }

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