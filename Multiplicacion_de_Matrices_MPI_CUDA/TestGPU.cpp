#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>

using namespace std;

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

void DisplayCUDAHeader(stringstream &stream);

int main(int argc, char *argv[]) {
  int numtasks, taskid, numworkers, mtype, rc, flag;

  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  printf("Number tasks: %d\n", numtasks);

  if (numtasks < 2) {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }

  numworkers = numtasks - 1;

  printf("Number workers: %d\n", numworkers);

  if (taskid == MASTER) {
    flag = 0;
    for (int dest = 1; dest <= numworkers; dest++) {
      printf("Checking task %d flag=%d\n", dest, flag);
      MPI_Send(&flag, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
    }

    for (int source = 1; source <= numworkers; source++) {
      string str;
      unsigned int size = 0;
      MPI_Recv(&flag, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
      MPI_Recv(&size, 1, MPI_UNSIGNED, source, FROM_WORKER, MPI_COMM_WORLD, &status);
      str.resize(size, '-');
      MPI_Recv(&str[0], size, MPI_CHAR, source, FROM_WORKER, MPI_COMM_WORLD, &status);
      printf("Received results from task %d, flag=%d\n", source, flag);
      cout << str << endl;
    }
  } else if (taskid > MASTER) {
    MPI_Recv(&flag, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);

    flag = 1;
    stringstream stream;

    char name[100];
    int resultlen = 0;
    MPI_Get_processor_name(name, &resultlen);

    stream << "=========================================================" << std::endl;
    stream << "Current Node: " << name << std::endl;
    DisplayCUDAHeader(stream);
    stream << "=========================================================" << std::endl;

    string str = stream.str();
    unsigned int size = str.size();

    MPI_Send(&flag, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    MPI_Send(&size, 1, MPI_UNSIGNED, MASTER, FROM_WORKER, MPI_COMM_WORLD);
    MPI_Send(&str[0], size, MPI_CHAR, MASTER, FROM_WORKER, MPI_COMM_WORLD);
  }

  MPI_Finalize();
}