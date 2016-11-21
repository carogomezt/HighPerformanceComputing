#include "mpi.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

void Mult_RowMat(float *M_a, float *M_b, float *R_c, int NRA, int NCA, int NCB);

#define MASTER 0      /* taskid of first task */
#define FROM_MASTER 1 /* setting a message type */
#define FROM_WORKER 2 /* setting a message type */

int main(int argc, char *argv[]) {
  int numtasks,   /* number of tasks in partition */
      taskid,     /* a task identifier */
      numworkers, /* number of worker tasks */
      source,     /* task id of message source */
      dest,       /* task id of message destination */
      mtype,      /* message type */
      elements,   /* elements of matrix A sent to each worker */
      averow, extra,
      offset,      /* used to determine elements sent to each worker */
      i, j, k, rc; /* misc */

  int NRA = 3;
  int NCA = 3;
  int NCB = 3;

  float *a, /* matrix A to be multiplied */
      *b,   /* matrix B to be multiplied */
      *c;   /* result matrix C */

  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  if (numtasks < 2) {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }

  numworkers = numtasks - 1;

  /**************************** master task
   * ************************************/
  if (taskid == MASTER) {
    a = (float *)malloc(NRA * NCA * sizeof(float));
    b = (float *)malloc(NCA * NCB * sizeof(float));
    c = (float *)malloc(NRA * NCB * sizeof(float));

    printf("mpi_mm has started with %d tasks.\n", numtasks);
    printf("Initializing arrays...\n");
    float cont = 0;
    for (i = 0; i < NRA; i++) {
      for (j = 0; j < NCA; j++) {
        a[i * NCA + j] = 1;
      }
    }

    cont = 0;
    for (i = 0; i < NCA; i++) {
      for (j = 0; j < NCB; j++) {
        b[i * NCB + j] = 1;
      }
    }

    /* Send matrix data to the worker tasks */
    averow = NRA / numworkers;
    extra = NRA % numworkers;
    offset = 0;
    mtype = FROM_MASTER;
    for (dest = 1; dest <= numworkers; dest++) {
      if (dest <= extra) {
        elements = averow + 1;
      } else {
        elements = averow;
      }
      printf("Sending %d elements to task %d offset=%d\n", elements, dest,
             offset);
      // Fila de inicio
      MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      // Numero de Filas
      MPI_Send(&elements, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      // Filas Matriz A
      MPI_Send(&a[offset * NCA], elements * NCA, MPI_FLOAT, dest, mtype,
               MPI_COMM_WORLD);
      // Matriz B
      MPI_Send(b, NCA * NCB, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
      offset = offset + elements;
    }

    /* Receive results from worker tasks */
    mtype = FROM_WORKER;
    for (i = 1; i <= numworkers; i++) {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&elements, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&c[offset * NCB], elements * NCB, MPI_FLOAT, source, mtype,
               MPI_COMM_WORLD, &status);
      printf("Received results from task %d\n", source);
    }

    /* Print results */
    printf("******************************************************\n");
    printf("Result Matrix:\n");

    for (i = 0; i < NRA; i++) {
      for (j = 0; j < NCB; j++) {
        cout << c[i * NCB + j] << " ";
      }
      cout << endl;
    }

    printf("\n******************************************************\n");
    printf("Done.\n");
  }

  /**************************** worker task
   * *************************************/
  if (taskid > MASTER) {
    mtype = FROM_MASTER;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&elements, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

    a = (float *)malloc(elements * NCA * sizeof(float));
    b = (float *)malloc(NCA * NCB * sizeof(float));
    c = (float *)malloc(elements * NCB * sizeof(float));

    MPI_Recv(a, elements * NCA, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD,
             &status);
    MPI_Recv(b, NCA * NCB, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);

    for (int k = 0; k < NCB; ++k) {
      for (int i = 0; i < elements; ++i) {
        for (int j = 0; j < NCA; ++j) {
          c[i * NCB + k] += a[i * NCA + k] * b[k * NCB + j];
        }
      }
    }

    // printf("MAT A\n");
    // for (int i = 0; i < NRA; i++) {
    //   for (int j = 0; j < NCA; j++){
    //     printf("%f ", a[i * NCA + j]);
    //   }
    //   printf("\n");
    // }

    // printf("MAT B\n");
    // for (int i = 0; i < NCA; i++) {
    //   for (int j = 0; j < NCB; j++){
    //     printf("%f ", b[i * NCB + j]);
    //   }
    //   printf("\n");
    // }

    // Mult_RowMat(a, b, c, elements, NCA, NCB);

    mtype = FROM_WORKER;
    MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(&elements, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    MPI_Send(c, elements * NCB, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);
  }
  MPI_Finalize();
  free(a);
  free(b);
  free(c);
}
