#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int chunk_size = atoi(argv[1]);
    int chunks = size/chunk_size;
    char command[1024];
    char command2[1024];
    // Create the command string using the provided arguments
    sprintf(command, "cp -a /lustre/orion/csc359/scratch/mschanen/git/milepost7/Milepost7Compiled /mnt/bb/mschanen > /dev/null 2>&1");
    sprintf(command2, "cp -a /lustre/orion/csc359/scratch/mschanen/git/milepost7/cases /mnt/bb/mschanen > /dev/null 2>&1");
    if (rank == 0) {
        printf("Total number of chunks %d\n", chunks);
    }
    for (int i = 0; i < chunks; i++) {
        if ((i * chunk_size) < rank < ((i+1) * chunk_size)) {

            // Execute the command
            int status = system(command);
            int status2 = system(command2);
        }
        if (rank == 0) {
            printf("Chunk %d\n", i);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
