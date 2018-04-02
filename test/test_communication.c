/*============================================================*/
/* Solve a 2D scalar wave equation                            */
/* MPI implementation                                         */ 
/* Soham 3/2018                                               */
/*============================================================*/

#include<mpi.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

#define nx 2      // no. of points in x
#define ny 2      // no. of points in y

int main(int argc, char *argv[]){
    double **uold,  **ucur;
    double *uold1D, *ucur1D;
    int i, j;
    int ixs, ixe, iys, iye, ixem, iyem;
    int gnx, gny, gnxgny;
    int nxnom, nynom, nprocs, procID;
    int testproc;

    MPI_Status status;                  // Status flag
    MPI_Datatype row_type, column_type; // for message passing 
    MPI_Request request[16];            // request handle array for non-blocking send and recieves 

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);
   
    // nominal number of points in each patch without ghost zones
    nxnom = nx/2; 
    nynom = ny/2;

    // local starting and ending indices (including ghost zones)
    ixs  = 0;
    ixe  = nxnom + 1;
    iys  = 0;
    iye  = nynom + 1;

    // ending index minus 1 (useful for exchanging ghost zones)
    ixem = ixe - 1;
    iyem = iye - 1;
    
    // Define array sizes including ghost zones (or nxnom + 2)
    gnx = nxnom + 2;
    gny = nynom + 2;
    gnxgny = gnx*gny;
    
    // Allocate memory dynamically for these arrays
    // This can be done differently if you use 1D arrays
    uold1D = malloc(gnxgny*sizeof(double*));
    ucur1D = malloc(gnxgny*sizeof(double*));
    uold   = malloc(gnx*sizeof(double*));
    ucur   = malloc(gnx*sizeof(double*));
    
    for (i=0; i<=gnx; i++){
        uold[i] = &uold1D[i*gny];
        ucur[i] = &ucur1D[i*gny];
    }

    // Initialize uold
    for (i=0; i<=ixe; i++){
        for (j=0; j<=iye; j++){
            uold[i][j] = (double)procID;
        }
    }

    testproc=3;

    if (procID==testproc){
    printf("------------------------------------\n");
    printf("proc[%i] before exchange\n", procID);
    printf("------------------------------------\n");
    for (i=0; i<=ixe; i++){
        for (j=0; j<=iye; j++){
            printf("%1.1f\t", uold[i][j]);
        }
        printf("\n");
    } 
    printf("------------------------------------\n");
    }

    // Define MPI datatypes for contiguous and non-contiguous data
    MPI_Type_contiguous(nynom, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);
    MPI_Type_vector(nxnom, 1, gny, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);
  
    // Initiate communication
    if(procID==0){
        // send and recieve the right edge
        MPI_Isend(&uold[1][iyem], 1, column_type, 1,  0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&uold[1][iye],  1, column_type, 1,  1, MPI_COMM_WORLD, &request[1]);
       
        // send and recieve the bottom edge
        MPI_Isend(&uold[ixem][1], 1, row_type,    2,  0, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&uold[ixe][1],  1, row_type,    2,  0, MPI_COMM_WORLD, &request[3]);
    
    } else if (procID==1){
        // send and recieve the left edge
        MPI_Irecv(&uold[1][0],    1, column_type, 0,  0, MPI_COMM_WORLD, &request[4]);
        MPI_Isend(&uold[1][1],    1, column_type, 0,  1, MPI_COMM_WORLD, &request[5]);
        
        // send and recieve the bottom edge
        MPI_Isend(&uold[ixem][1], 1, row_type,    3,  0, MPI_COMM_WORLD, &request[6]);
        MPI_Irecv(&uold[ixe][1],  1, row_type,    3,  0, MPI_COMM_WORLD, &request[7]);

    } else if (procID==2){
        // send and recieve the right edge
        MPI_Isend(&uold[1][iyem], 1, column_type, 3,  0, MPI_COMM_WORLD, &request[8]);
        MPI_Irecv(&uold[1][iye],  1, column_type, 3,  1, MPI_COMM_WORLD, &request[9]);
        
        // send and recieve the top edge
        MPI_Isend(&uold[1][1],    1, row_type,    0,  0, MPI_COMM_WORLD, &request[10]);
        MPI_Irecv(&uold[0][1],    1, row_type,    0,  0, MPI_COMM_WORLD, &request[11]);
    
    } else {
        // send and recieve the left edge
        MPI_Irecv(&uold[1][0],    1, column_type, 2,  0, MPI_COMM_WORLD, &request[12]);
        MPI_Isend(&uold[1][1],    1, column_type, 2,  1, MPI_COMM_WORLD, &request[13]);
        
        // send and recieve the top edge
        MPI_Isend(&uold[1][1],    1, row_type,    1,  0, MPI_COMM_WORLD, &request[14]);
        MPI_Irecv(&uold[0][1],    1, row_type,    1,  0, MPI_COMM_WORLD, &request[15]);
    }


    // Make sure all non-blocking messages have arrive for each process
    for (i=procID*4; i<=(procID*4)+3; i++){
        MPI_Wait(&request[i], &status);
    }
    
    if (procID==testproc){
    printf("proc[%i] after exchange\n", procID);
    printf("------------------------------------\n");
    for (i=0; i<=ixe; i++){
        for (j=0; j<=iye; j++){
            printf("%1.1f\t", uold[i][j]);
        }
        printf("\n");
    } 
    printf("------------------------------------\n");
    }

    MPI_Finalize();
}
