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

int main(int argc, char *argv[]){
    double **uold,  **ucur;
    double *uold1D, *ucur1D;
    int i, j, li, lj, l, m;
    int gixs, gixe, giys, giye;
    int ixs, ixe, iys, iye, ixem, iyem;
    int gnx, gny, gnxgny;
    int nxnom, nynom, nprocs, procID;
    int nxproc, nyproc;
    double x, y, dx, dy, sum, sumloc;
    double wx[nx], wy[ny];

    MPI_Datatype  mem_type, file_type;
    MPI_Comm comm_cart;
    MPI_File fh;
    FILE *fp;

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
   
    // global indices without ghost zones
    gixs = ((procID/nxproc)*nxnom) + 1;  
    gixe = gixs + nxnom - 1;             
    giys = ((procID%nyproc)*nynom) + 1;  
    giye = giys + nynom - 1;             
    
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

    // Set dx and dy
    dx = 2.0/(nx-1.0);
    dy = 2.0/(ny-1.0);

    // Initialize uold
    for (i=0; i<=ixe; i++){
        for (j=0; j<=iye; j++){
            uold[i][j] = 0.0;
        }
    }

    // Re-initialize uold
    for (i=1; i<=nx; i++){
        for (j=1; j<=ny; j++){
            x = (double)(1 + nx - 2*i)/(double)(1 - nx); 
            y = (double)(1 + ny - 2*j)/(double)(1 - ny);
            if (i >= gixs && i <= gixe && j >= giys && j <= giye){
                li = i - gixs + 1;   
                lj = j - giys + 1;   
                uold[li][lj] = x*x + y*y;      
            }
        }
    }
   
    if(1){
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

    /* create user-defined types for parallel i/o */
    size[0] = nx;
    size[1] = ny;
    subsize[0] = nxnom;
    subsize[1] = nynom;
    MPI_Cart_coords(comm_cart, myid, 2, start);

    start[0] *= nxnom;
    start[1] *= nynom;
    MPI_Type_create_subarray(2, size, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &file_type);
    MPI_Type_commit(&file_type);

    size[0] = nx;
    size[1] = ny;
    subsize[0] = nxnom;
    subsize[1] = nynom;
    start[0] = 1;
    start[1] = 1;
    MPI_Type_create_subarray(2, size, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &mem_type);
    MPI_Type_commit(&mem_type);

    /* open file for MPI-2 parallel i/o */
    MPI_File_open(MPI_COMM_WORLD, "final.dat", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_DOUBLE, file_type, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, &new[0][0], 1, mem_type, &status);
    MPI_File_close(&fh);
 
    MPI_Finalize();
}
