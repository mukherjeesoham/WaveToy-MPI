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

#define nx 200      // no. of points in x
#define ny 200      // no. of points in y

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

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);

    // number of processors in each direction
    nxproc = 4;
    nyproc = 4;

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
   
    if(0){
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

    // Compute integration weights
    for (i=1; i<=nx; i++){
        if (i==1 || i==nx){
            wx[i] = 0.5;
        } else {
            wx[i] = 1.0;
        }
    }
    
    for (j=1; j<=ny; j++){
        if (j==1 || j==ny){
            wy[j] = 0.5;
        } else {
            wy[j] = 1.0;
        }
    }

    // Integrate on a single patch
    sum = 0.0;
    for (i=1; i<=nx; i++){
        for (j=1; j<=ny; j++){
            if (i >= gixs && i <= gixe && j >= giys && j <= giye){
                li = i - gixs + 1;   
                lj = j - giys + 1;   
                sum = sum + dx*dy*wx[i]*wy[j]*uold[li][lj];      
            }
        }
    }
    
    // Return sum with MPI Reduce
    sumloc = sum;
    MPI_Reduce(&sumloc, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (procID==0){
        printf("proc[%i] Total sum = %g\n", procID, sum);
    }
    
    MPI_Finalize();
}
