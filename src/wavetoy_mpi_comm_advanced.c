/*============================================================*/
/* Solve a 2D scalar wave equation                            */
/* MPI implementation where each MPI procress does the 
 * same thing. Introduce communication
 * where the domain is split vertically to allow for 
 * non-contigous data being sent. Only works for nprocs = 2   */
/* Soham 3/2018                                               */
/*============================================================*/

#include<mpi.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

#define nx 10      // no. of points in x
#define ny 10      // no. of points in y
#define nsteps 10     // no. of time steps

int main(int argc, char *argv[]){
    int **uold, **unew, **ucur;
    int *uold1D, *unew1D, *ucur1D;
    int nprocs, worker;
    double c, csq, x, y, u, dudt, dx, dy, dt, dtdx, dtdxsq;
    MPI_Status status;
    MPI_Datatype column_type;
    int nxprocs, nyprocs, nxnom, nynom;
    int i, j, k;
    int gixs, gixe, giys, giye;
    int ixs, ixe, iys, iye;
    int ixem, iyem;
    int gnx, gny, gnxgny;
    int li, lj;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &worker);

    // Domain decomposition (no. of workers)
    nxprocs = 1;           // Workers in x direction
    nyprocs = nprocs;                // Workers in y direction
    
    // nominal number of points in each patch
    // wihtout ghost zones
    nxnom = nx/nxprocs; 
    nynom = ny/nyprocs;

    // global starting and ending indices (withtout ghost zones)
    gixs = 1;                   // starting index for $worker along y
    gixe = nx;                  // ending index for $worker along y
    giys = (worker*nynom) + 1;  // starting index for $worker along x 
    giye = giys + nynom - 1;    // ending index for $worker along x

    // local starting and ending indices (including ghost zones)
    ixs  = 0;
    ixe  = nxnom + 1;
    iys  = 0;
    iye  = nynom + 1;

    // ending index minus 1 (useful for exchanging ghost zones)
    ixem = ixe - 1;
    iyem = iye - 1;

    // Some basic quantities 
    c  = 1.0;                       // wave speed
    dt = 1.0/(double)(nsteps-1);    // final time is 1
    dx = 2.0/(double)(nx+1);        // x spans from -1 to 1
    dy = 2.0/(double)(ny+1);        // y spans from -1 to 1
    csq = c*c;  
    dtdx = dt/dx;
    dtdxsq = dtdx*dtdx;
    
    // Define array sizes including ghost zones (or nxnom + 2)
    gnx = ixe - ixs + 1;
    gny = iye - iys + 1;
    gnxgny = gnx*gny;
    
    // Allocate memory dynamically for these arrays
    // This can be done differently if you use 1D arrays
    uold1D = malloc(gnxgny*sizeof(double*));
    ucur1D = malloc(gnxgny*sizeof(double*));
    unew1D = malloc(gnxgny*sizeof(double*));
    uold   = malloc(gnx*sizeof(double*));
    ucur   = malloc(gnx*sizeof(double*));
    unew   = malloc(gnx*sizeof(double*));
    
    for (i=0; i<=gnx; i++){
        uold[i] = &uold1D[i*gny];
        ucur[i] = &ucur1D[i*gny];
        unew[i] = &unew1D[i*gny];
    }

    // Initialize uold
    for (i=1; i<=nx; i++){
        for (j=1; j<=ny; j++){
            x = -(double)(1 + nx + 2*i)/(double)(1 + nx); 
            y = -(double)(1 + ny + 2*j)/(double)(1 + ny);
            u = exp(-x*x/0.01 - y*y/0.01);                  // Initial condition: Gaussian
            // check which processor this point belongs to 
            // and set the point if and only if it belongs to the current processor.
            if (i >= gixs && i <= gixe){
                li = i - gixs + 1;   
                lj = j;
                uold[li][lj] = u;
            }
        }
    }
 
  
   printf("u[1][0] = %g\n", u[1][0]);

   // Initialize ucur 
   // For this, only the inner boundaries need to be exchanged
   // Define MPI datatype
   
   MPI_Type_vector(nx, 1, gny, MPI_DOUBLE, &column_type);
   MPI_Type_commit(&column_type);
    
   if (worker==0){     // Currently works for only two processes 
       MPI_Send(&uold[0][iyem], 1, column_type, 1, 0, MPI_COMM_WORLD);
       MPI_Recv(&uold[0][iye],  1, column_type, 1, 1, MPI_COMM_WORLD, &status);
   } else{            
       MPI_Recv(&uold[1][0],    1, column_type, 0, 0, MPI_COMM_WORLD, &status);
       MPI_Send(&uold[1][1],    1, column_type, 0, 1, MPI_COMM_WORLD);
   }
   
   for (i=1; i<=nxnom; i++){
       for (j=1; j<=nynom; j++){   
           dudt = 0.0; 
           ucur[i][j] = uold[i][j] - 2.0*dt*dudt 
                    + 0.5*csq*dtdxsq*(uold[i+1][j] - 2.0*uold[i][j] + uold[i-1][j]
                                    + uold[i][j+1] - 2.0*uold[i][j] + uold[i][j-1]);    
       }
   }
   
   if (worker ==0){
       printf("Jetzt geht's los!\n");
   }

   // update solution to next step for n steps
   for (k=0; k<=nsteps; k++){
      if (worker == 0){
          // Impose Dirichlet BCs on the corners
          uold[0][0]   = 0.0;
          ucur[0][0]   = 0.0;
          uold[0][iye] = 0.0;
          ucur[0][iye] = 0.0;
          uold[ixe][0] = 0.0;
          ucur[ixe][0] = 0.0;
          uold[ixe][iye] = 0.0;
          ucur[ixe][iye] = 0.0;

          // Impose Dirichlet BCs on the top
          for (j=1; j<=iyem; j++){
              uold[0][j] = 0.0;
              ucur[0][j] = 0.0;
          }

          // Impose Dirichlet BCs on the bottom
          for (j=1; j<=iyem; j++){
              uold[ixe][j] = 0.0;
              ucur[ixe][j] = 0.0;
          }
          
          // Impose Dirichlet BCs on the left
          for (i=1; i<=ixem; i++){
              uold[i][0] = 0.0;
              ucur[i][0] = 0.0;
          }
          
      } else{
          // Impose Dirichlet BCs on the corners
          uold[ixe][0] = 0.0;
          ucur[ixe][0] = 0.0;
          uold[ixe][iye] = 0.0;
          ucur[ixe][iye] = 0.0;
          uold[0][0] = 0.0;
          ucur[0][0] = 0.0;
          uold[0][iye] = 0.0;
          ucur[0][iye] = 0.0;

          
          // Impose Dirichlet BCs on the top
          for (i=1; i<=iyem; i++){
              uold[0][i] = 0.0;
              ucur[0][i] = 0.0;
          }

          // Impose Dirichlet BCs on the bottom
          for (i=1; i<=iyem; i++){
              uold[ixe][i] = 0.0;
              ucur[ixe][i] = 0.0;
          }
          // Impose Dirichlet BCs on the right
          for (i=1; i<=ixem; i++){
              uold[i][iye] = 0.0;
              ucur[i][iye] = 0.0;
          }
      }
       
      // Exchange data between the patches
      if (worker==0){     // Currently works for only two processes 
          MPI_Send(&uold[0][iyem], 1, column_type, 1, 0, MPI_COMM_WORLD);
          MPI_Recv(&uold[0][iye],  1, column_type, 1, 1, MPI_COMM_WORLD, &status);
          MPI_Send(&ucur[0][iyem], 1, column_type, 1, 0, MPI_COMM_WORLD);
          MPI_Recv(&ucur[0][iye],  1, column_type, 1, 1, MPI_COMM_WORLD, &status);
      } else{            
          MPI_Recv(&uold[1][0],    1, column_type, 0, 0, MPI_COMM_WORLD, &status);
          MPI_Send(&uold[1][1],    1, column_type, 0, 1, MPI_COMM_WORLD);
          MPI_Recv(&ucur[1][0],    1, column_type, 0, 0, MPI_COMM_WORLD, &status);
          MPI_Send(&ucur[1][1],    1, column_type, 0, 1, MPI_COMM_WORLD);
      }

      // compute the next time level
      for (i=1; i<=nx; i++){
          for (j=1; j<=nynom; j++){
              unew[i][j] = 2*ucur[i][j] + uold[i][j] 
                           + csq*dtdxsq*(ucur[i+1][j] - 2.0*ucur[i][j] + ucur[i-1][j]
                                       + ucur[i][j+1] - 2.0*ucur[i][j] + ucur[i][j-1]);
          }
      }

      // update the arrays
      for (i=1; i<=nx; i++){
          for (j=1; j<=nynom; j++){
              uold[i][j] = ucur[i][j];
              ucur[i][j] = unew[i][j];
          }
      }
   }

   
   printf("Process %i of %i finished.\n", worker+1, nprocs);
   MPI_Finalize();
}
