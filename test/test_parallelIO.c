/*============================================================*/
/* Solve a 2D scalar wave equation                            */
/* MPI implementation                                         */
/* Generalize to n processes in each direction                */
/* Soham 3/2018                                               */
/*============================================================*/

#include<mpi.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

#define nx 2      // no. of points in x
#define ny 2      // no. of points in y

int sub2ind(int ix, int iy, int nxproc, int nyproc){
    return iy + nxproc*ix;
}

int main(int argc, char *argv[]){
    double **uold,  **ucur;
    double *uold1D, *ucur1D;
    int i, j;
    int ixs, ixe, iys, iye, ixem, iyem;
    int gnx, gny, gnxgny;
    int nxnom, nynom, nprocs, procID;
    int cartID[2];
    int nxproc, nyproc, testproc;
    MPI_Status status;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);

    // number of processors in each direction
    nxproc = 2;
    nyproc = 2;

    // nominal number of points in each patch without ghost zones
    nxnom = nx/nxproc;
    nynom = ny/nyproc;

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

    // Set up parallel I/O
    // create user-defined types for parallel i/o
      
    size[0]    = nx;
    size[1]    = ny;
    subsize[0] = nxnom;
    subsize[1] = nynom;

    MPI_Cart_coords(comm_cart, myid, 2, start);

    start[0] *= nxnom;
    start[1] *= nynom;

    MPI_Type_create_subarray(2, size, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &file_type);
    MPI_Type_commit(&file_type);

    size[0]    = gnx;
    size[1]    = gny;
    subsize[0] = nxnom;
    subsize[1] = nynom;
    start[0]   = 1;
    start[1]   = 1;
    
    MPI_Type_create_subarray(2, size, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &mem_type);
    MPI_Type_commit(&mem_type);

    // open file for MPI-2 parallel I/O
    MPI_File_open(MPI_COMM_WORLD, "output.dat", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_DOUBLE, file_type, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, &new[0][0], 1, mem_type, &status);
    MPI_File_close(&fh);
    
    MPI_Finalize();
}
