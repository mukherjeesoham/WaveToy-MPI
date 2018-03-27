/*============================================================*/
/* Solve a 2D scalar wave equation                            */
/* MPI implementation                                         */ 
/* Soham 3/2018                                               */
/*============================================================*/

#include<mpi.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

#define nx 10      // no. of points in x
#define ny 10      // no. of points in y
#define nt 10      // no. of time steps

int main(int argc, char *argv[]){
    double **uold, **unew, **ucur;
    double *uold1D, *unew1D, *ucur1D;
    int i, j, k, li, lj;
    int gixs, gixe, giys, giye;
    int ixs, ixe, iys, iye;
    int ixem, iyem;
    int gnx, gny, gnxgny;
    double c, csq, x, y, u, dudt, dudx, dudy, dx, dy, dt, dtdx, dtdxsq;
    int nprocs, procID, cartID, ndims, nxnom, nynom;
    int reorder, nsdir, ewdir, proc_north, proc_south, proc_east, proc_west, proc_corner; 
    int dimsize[2], periods[2], size[2], subsize[2], start[2];
    double E, sumE;

    MPI_Status status;
    MPI_Datatype row_type, column_type; // for message passing 
    MPI_Datatype mem_type, file_type;   // for paralel I/O
    MPI_Request request[12];            // request handle array for non-blocking send and recieves 
    MPI_Comm comm_cart;                 // Communicator handle for the virtual topology
    MPI_File fh;                        // for parallel I/O

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);

    // Domain decomposition
    // We want to run on 4 processes, 2 in each direction
    ndims = 2;
    dimsize[0] = nprocs/ndims;      // no. of processes in x direction
    dimsize[1] = nprocs/ndims;      // no. of processes in y direction
    
    // Create a virtual topology for 4 procs
    nsdir = 0;          // Direction label for north-south
    ewdir = 1;          // Direction label for east-west
    periods[0] = 0;     // not periodic in x
    periods[1] = 0;     // not periodix in y
    reorder    = 0;     // do not allow MPI to reoder the ranks of the processes.
   
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dimsize, periods, reorder, &comm_cart); // Create a new communicator with Cartesian Topology.
    MPI_Cart_shift(comm_cart, ewdir,  1, &cartID, &proc_east);   
    MPI_Cart_shift(comm_cart, ewdir, -1, &cartID, &proc_west);  
    MPI_Cart_shift(comm_cart, nsdir, -1, &cartID, &proc_north); 
    MPI_Cart_shift(comm_cart, nsdir,  1, &cartID, &proc_south); 

    // now define the processors to exchange data at the corners
    switch(procID){
        case 0:
            proc_corner = 3;
            break;
        case 1:
            proc_corner = 2;  
            break;
        case 2:
            proc_corner = 1; 
            break;
        case 3:
            proc_corner = 0;
            break;
    }

    // nominal number of points in each patch without ghost zones
    nxnom = nx/dimsize[0]; 
    nynom = ny/dimsize[1];

    // global starting and ending indices (withtout ghost zones)
    gixs = ((procID/2)*nxnom) + 1;  // starting index for worker along x
    gixe = gixs + nxnom - 1;        // ending index for worker along x
    giys = ((procID%2)*nynom) + 1;  // starting index for worker along y 
    giye = giys + nynom - 1;        // ending index for worker along y

    // local starting and ending indices (including ghost zones)
    ixs  = 0;
    ixe  = nxnom + 1;
    iys  = 0;
    iye  = nynom + 1;

    // ending index minus 1 (useful for exchanging ghost zones)
    ixem = ixe - 1;
    iyem = iye - 1;
    
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

    // Some basic quantities for the wave equation
    c  = 1.0;                       // wave speed
    dt = 1.0/(double)(nt+1);    // final time is 1
    dx = 2.0/(double)(nx+1);        // x spans from -1 to 1
    dy = 2.0/(double)(ny+1);        // y spans from -1 to 1
    csq = c*c;  
    dtdx = dt/dx;
    dtdxsq = dtdx*dtdx;

    // Initialize uold
    for (i=0; i<=nx+1; i++){
        for (j=0; j<=ny+1; j++){
            if (i==0 || j == 0 || i == nx+1 || j == ny+1){              // set the boundaries of uold
                if (i >= gixs && i <= gixe && j >= giys && j <= giye){  // only if it belongs to the current processor
                    li = i - gixs + 1;   
                    lj = j - giys + 1;   
                    uold[i][j] = 0.0;
                }   
            }else{                                                      // set the interior points of uold
                x = (double)(1 + nx - 2*i)/(double)(-2 + 1 - nx); 
                y = (double)(1 + ny - 2*j)/(double)(-2 + 1 - ny);
                if (i >= gixs && i <= gixe && j >= giys && j <= giye){  // only if it belongs to the current processor
                    li = i - gixs + 1;   
                    lj = j - giys + 1;   
                    uold[li][lj] = 100.0*exp(-x*x/0.01 - y*y/0.01);      
                }
            }
        }
    }

   // Define MPI datatypes for contiguous and non-contiguous data
   MPI_Type_vector(nxnom, 1, gny, MPI_DOUBLE, &column_type);
   MPI_Type_commit(&column_type);
   MPI_Type_contiguous(nynom, MPI_DOUBLE, &row_type);
   MPI_Type_commit(&row_type);
   
   // Initialize ucur
   switch(procID){
       case 0:
           // send and recieve the left edge
           MPI_Isend(&uold[1][iyem], 1, column_type, proc_east,  1112, comm_cart, &request[0]); 
           MPI_Irecv(&uold[1][iye],  1, column_type, proc_east,  1211, comm_cart, &request[1]); 
           
           // send and recieve the bottom edge
           MPI_Isend(&uold[ixem][1], 1, row_type,    proc_south, 1121, comm_cart, &request[2]);
           MPI_Irecv(&uold[ixe][1],  1, row_type,    proc_south, 2111, comm_cart, &request[3]); 
          
           // send and recieve the corner
           MPI_Isend(&uold[ixem][iyem],  1, MPI_DOUBLE, proc_corner, 1122, MPI_COMM_WORLD, &request[4]);
           MPI_Irecv(&uold[ixe][iye],    1, MPI_DOUBLE, proc_corner, 2211, MPI_COMM_WORLD, &request[5]);
           break;
       case 1:
           // send and recieve the right edge
           MPI_Isend(&uold[1][1],    1, column_type, proc_west,  1211, comm_cart, &request[0]); 
           MPI_Irecv(&uold[1][0],    1, column_type, proc_west,  1112, comm_cart, &request[1]); 
           
           // send and recieve the bottom edge
           MPI_Isend(&uold[ixem][1], 1, row_type,    proc_south, 1222, comm_cart, &request[2]);
           MPI_Irecv(&uold[ixe][1],  1, row_type,    proc_south, 2212, comm_cart, &request[3]); 
          
           // send and recieve the corner
           MPI_Isend(&uold[ixem][1], 1, MPI_DOUBLE,  proc_corner, 1221, MPI_COMM_WORLD, &request[4]);
           MPI_Irecv(&uold[ixe][0],  1, MPI_DOUBLE,  proc_corner, 2112, MPI_COMM_WORLD, &request[5]);
           break;
       case 2:
           // send and recieve the left edge
           MPI_Isend(&uold[1][iyem], 1, column_type, proc_east,  1112, comm_cart, &request[0]); 
           MPI_Irecv(&uold[1][iye],  1, column_type, proc_east,  1211, comm_cart, &request[1]); 
           
           // send and recieve the top edge
           MPI_Isend(&uold[1][1],    1, row_type,    proc_north, 2111, comm_cart, &request[2]);
           MPI_Irecv(&uold[0][1],    1, row_type,    proc_north, 1121, comm_cart, &request[3]); 
          
           // send and recieve the corner
           MPI_Isend(&uold[1][iyem], 1, MPI_DOUBLE, proc_corner, 1122, MPI_COMM_WORLD, &request[4]);
           MPI_Irecv(&uold[0][iye],  1, MPI_DOUBLE, proc_corner, 2211, MPI_COMM_WORLD, &request[5]);
           break;
       case 3:
           // send and recieve the right edge
           MPI_Isend(&uold[1][1],    1, column_type, proc_west,  2221, comm_cart, &request[0]); 
           MPI_Irecv(&uold[1][0],    1, column_type, proc_west,  2122, comm_cart, &request[1]); 
           
           // send and recieve the top edge
           MPI_Isend(&uold[1][1],    1, row_type,    proc_south, 2212, comm_cart, &request[2]);
           MPI_Irecv(&uold[0][1],    1, row_type,    proc_south, 1222, comm_cart, &request[3]); 
          
           // send and recieve the corner
           MPI_Isend(&uold[1][1],    1, MPI_DOUBLE,  proc_corner, 2211, MPI_COMM_WORLD, &request[4]);
           MPI_Irecv(&uold[0][0],    1, MPI_DOUBLE,  proc_corner, 1122, MPI_COMM_WORLD, &request[5]);
           break;
   }

   // Make sure all non-blocking messages have arrived
   for (i=0; i<=5; i++){
       MPI_Wait(&request[i], &status);
   } 
   
   // Compute ucur
   for (i=1; i<=nxnom; i++){
       for (j=1; j<=nynom; j++){   
           dudt = 0.0; 
           ucur[i][j] = uold[i][j] - 2.0*dt*dudt 
                    + 0.5*csq*dtdxsq*(uold[i+1][j] - 2.0*uold[i][j] + uold[i-1][j]
                                    + uold[i][j+1] - 2.0*uold[i][j] + uold[i][j-1]);    
       }
   }
   
   if (procID == 0){
       printf("Jetzt geht's los!\n");
   }

   // update solution to next step for n steps
   for (k=0; k<=nt; k++){
       
      // Exchange data between the patches and set boundary conditions
      switch(procID){
          case 0:
              // set Dirichlet BCs nw, ne, sw corners
              uold[0][0]   = 0.0;
              ucur[0][0]   = 0.0;
              uold[0][iye] = 0.0;
              ucur[0][iye] = 0.0;
              uold[ixe][0] = 0.0;
              ucur[ixe][0] = 0.0;

              // set Dirichlet BCs on top edge
              for (j=1; j<=iyem; j++){
                  uold[0][j] = 0.0;
                  ucur[0][j] = 0.0;
              }

              // set Dirichlet BCs on left edge
              for (i=1; i<=ixem; j++){
                  uold[i][0] = 0.0;
                  ucur[i][0] = 0.0;
              }

              // send and recieve the right edge
              MPI_Isend(&uold[1][iyem], 1, column_type, proc_east,  1112, comm_cart, &request[0]); 
              MPI_Irecv(&uold[1][iye],  1, column_type, proc_east,  1211, comm_cart, &request[1]); 
              MPI_Isend(&ucur[1][iyem], 1, column_type, proc_east,  1112, comm_cart, &request[2]); 
              MPI_Irecv(&ucur[1][iye],  1, column_type, proc_east,  1211, comm_cart, &request[3]); 
              
              // send and recieve the bottom edge
              MPI_Isend(&uold[ixem][1], 1, row_type,    proc_south, 1121, comm_cart, &request[4]);
              MPI_Irecv(&uold[ixe][1],  1, row_type,    proc_south, 2111, comm_cart, &request[5]); 
              MPI_Isend(&ucur[ixem][1], 1, row_type,    proc_south, 1121, comm_cart, &request[6]);
              MPI_Irecv(&ucur[ixe][1],  1, row_type,    proc_south, 2111, comm_cart, &request[7]); 
             
              // send and recieve the corner
              MPI_Isend(&uold[ixem][iyem],  1, MPI_DOUBLE, proc_corner, 1122, MPI_COMM_WORLD, &request[8]);
              MPI_Irecv(&uold[ixe][iye],    1, MPI_DOUBLE, proc_corner, 2211, MPI_COMM_WORLD, &request[9]);
              MPI_Isend(&ucur[ixem][iyem],  1, MPI_DOUBLE, proc_corner, 1122, MPI_COMM_WORLD, &request[10]);
              MPI_Irecv(&ucur[ixe][iye],    1, MPI_DOUBLE, proc_corner, 2211, MPI_COMM_WORLD, &request[11]);
              break;
          case 1:
              // set Dirichlet BCs nw, ne, se corners
              uold[0][0]     = 0.0;
              ucur[0][0]     = 0.0;
              uold[0][iye]   = 0.0;
              ucur[0][iye]   = 0.0;
              uold[ixe][iye] = 0.0;
              ucur[ixe][iye] = 0.0;

              // set Dirichlet BCs on top edge
              for (j=1; j<=iyem; j++){
                  uold[0][j] = 0.0;
                  ucur[0][j] = 0.0;
              }

              // set Dirichlet BCs on right edge
              for (i=1; i<=ixem; j++){
                  uold[i][iye] = 0.0;
                  ucur[i][iye] = 0.0;
              }

              // send and recieve the left edge
              MPI_Isend(&uold[1][1],    1, column_type, proc_west,  1211, comm_cart, &request[0]); 
              MPI_Irecv(&uold[1][0],    1, column_type, proc_west,  1112, comm_cart, &request[1]); 
              MPI_Isend(&ucur[1][1],    1, column_type, proc_west,  1211, comm_cart, &request[2]); 
              MPI_Irecv(&ucur[1][0],    1, column_type, proc_west,  1112, comm_cart, &request[3]); 
              
              // send and recieve the bottom edge
              MPI_Isend(&uold[ixem][1], 1, row_type,    proc_south, 1222, comm_cart, &request[4]);
              MPI_Irecv(&uold[ixe][1],  1, row_type,    proc_south, 2212, comm_cart, &request[5]); 
              MPI_Isend(&ucur[ixem][1], 1, row_type,    proc_south, 1222, comm_cart, &request[6]);
              MPI_Irecv(&ucur[ixe][1],  1, row_type,    proc_south, 2212, comm_cart, &request[7]); 
             
              // send and recieve the corner
              MPI_Isend(&uold[ixem][1], 1, MPI_DOUBLE,  proc_corner, 1221, MPI_COMM_WORLD, &request[8]);
              MPI_Irecv(&uold[ixe][0],  1, MPI_DOUBLE,  proc_corner, 2112, MPI_COMM_WORLD, &request[9]);
              MPI_Isend(&ucur[ixem][1], 1, MPI_DOUBLE,  proc_corner, 1221, MPI_COMM_WORLD, &request[10]);
              MPI_Irecv(&ucur[ixe][0],  1, MPI_DOUBLE,  proc_corner, 2112, MPI_COMM_WORLD, &request[11]);
              break;
          case 2:
              // set Dirichlet BCs sw, se, nw corners
              uold[0][0]     = 0.0;
              ucur[0][0]     = 0.0;
              uold[ixe][0]   = 0.0;
              ucur[ixe][0]   = 0.0;
              uold[ixe][iye] = 0.0;
              ucur[ixe][iye] = 0.0;

              // set Dirichlet BCs on bottom edge
              for (j=1; j<=iyem; j++){
                  uold[ixe][j] = 0.0;
                  ucur[ixe][j] = 0.0;
              }

              // set Dirichlet BCs on left edge
              for (i=1; i<=ixem; j++){
                  uold[i][0] = 0.0;
                  ucur[i][0] = 0.0;
              }

              // send and recieve the right edge
              MPI_Isend(&uold[1][iyem], 1, column_type, proc_east,  1112, comm_cart, &request[0]); 
              MPI_Irecv(&uold[1][iye],  1, column_type, proc_east,  1211, comm_cart, &request[1]); 
              MPI_Isend(&ucur[1][iyem], 1, column_type, proc_east,  1112, comm_cart, &request[2]); 
              MPI_Irecv(&ucur[1][iye],  1, column_type, proc_east,  1211, comm_cart, &request[3]); 
              
              // send and recieve the top edge
              MPI_Isend(&uold[1][1],    1, row_type,    proc_north, 2111, comm_cart, &request[4]);
              MPI_Irecv(&uold[0][1],    1, row_type,    proc_north, 1121, comm_cart, &request[5]); 
              MPI_Isend(&ucur[1][1],    1, row_type,    proc_north, 2111, comm_cart, &request[6]);
              MPI_Irecv(&ucur[0][1],    1, row_type,    proc_north, 1121, comm_cart, &request[7]); 
             
              // send and recieve the corner
              MPI_Isend(&uold[1][iyem], 1, MPI_DOUBLE, proc_corner, 1122, MPI_COMM_WORLD, &request[8]);
              MPI_Irecv(&uold[0][iye],  1, MPI_DOUBLE, proc_corner, 2211, MPI_COMM_WORLD, &request[9]);
              MPI_Isend(&ucur[1][iyem], 1, MPI_DOUBLE, proc_corner, 1122, MPI_COMM_WORLD, &request[10]);
              MPI_Irecv(&ucur[0][iye],  1, MPI_DOUBLE, proc_corner, 2211, MPI_COMM_WORLD, &request[11]);
              break;
          case 3:
              // set Dirichlet BCs sw, se, ne corners
              uold[0][iye]   = 0.0;
              ucur[0][iye]   = 0.0;
              uold[ixe][0]   = 0.0;
              ucur[ixe][0]   = 0.0;
              uold[ixe][iye] = 0.0;
              ucur[ixe][iye] = 0.0;

              // set Dirichlet BCs on bottom edge
              for (j=1; j<=iyem; j++){
                  uold[ixe][j] = 0.0;
                  ucur[ixe][j] = 0.0;
              }

              // set Dirichlet BCs on right edge
              for (i=1; i<=ixem; j++){
                  uold[i][iye] = 0.0;
                  ucur[i][iye] = 0.0;
              }
              
              // send and recieve the right edge
              MPI_Isend(&uold[1][1],    1, column_type, proc_west,  2221, comm_cart, &request[0]); 
              MPI_Irecv(&uold[1][0],    1, column_type, proc_west,  2122, comm_cart, &request[1]); 
              MPI_Isend(&ucur[1][1],    1, column_type, proc_west,  2221, comm_cart, &request[2]); 
              MPI_Irecv(&ucur[1][0],    1, column_type, proc_west,  2122, comm_cart, &request[3]); 
              
              // send and recieve the top edge
              MPI_Isend(&uold[1][1],    1, row_type,    proc_south, 2212, comm_cart, &request[4]);
              MPI_Irecv(&uold[0][1],    1, row_type,    proc_south, 1222, comm_cart, &request[5]); 
              MPI_Isend(&ucur[1][1],    1, row_type,    proc_south, 2212, comm_cart, &request[6]);
              MPI_Irecv(&ucur[0][1],    1, row_type,    proc_south, 1222, comm_cart, &request[7]); 
             
              // send and recieve the corner
              MPI_Isend(&uold[1][1],    1, MPI_DOUBLE,  proc_corner, 2211, MPI_COMM_WORLD, &request[8]);
              MPI_Irecv(&uold[0][0],    1, MPI_DOUBLE,  proc_corner, 1122, MPI_COMM_WORLD, &request[9]);
              MPI_Isend(&ucur[1][1],    1, MPI_DOUBLE,  proc_corner, 2211, MPI_COMM_WORLD, &request[10]);
              MPI_Irecv(&ucur[0][0],    1, MPI_DOUBLE,  proc_corner, 1122, MPI_COMM_WORLD, &request[11]);
              break;
      }
      
      // Make sure all non-blocking messages have arrived
      for (i=0; i<=11; i++){
          MPI_Wait(&request[i], &status);
      } 
   
      // compute the next time level
      for (i=1; i<=nx; i++){
          for (j=1; j<=nynom; j++){
              unew[i][j] = 2*ucur[i][j] + uold[i][j] 
                           + csq*dtdxsq*(ucur[i+1][j] - 2.0*ucur[i][j] + ucur[i-1][j]
                                       + ucur[i][j+1] - 2.0*ucur[i][j] + ucur[i][j-1]);
          }
      }

      // Compute the energy at current patch
      E = 0.0;
      for (i=1; i<=nx; i++){
          for (j=1; j<=ny; j++){
              dudx = ucur[i+1][j] - 2.0*ucur[i][j] + ucur[i-1][j];
              dudy = ucur[i][j+1] - 2.0*ucur[i][j] + ucur[i][j-1];
              dudt = unew[i][j]   - 2.0*ucur[i][j] + uold[i][j];
              E = E + dudx*dudx + dudy*dudy + dudt*dudt; 
          }
      }

      // Compute the energy at the current time level
      sumE = E;
      MPI_Reduce(&sumE, &E, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      // update the arrays
      for (i=1; i<=nx; i++){
          for (j=1; j<=nynom; j++){
              uold[i][j] = ucur[i][j];
              ucur[i][j] = unew[i][j];
          }
      }
   
      // Create custom defined types for paralel I/O
      size[0] = nx;
      size[1] = ny;
      subsize[0] = nxnom;
      subsize[1] = nynom;

      MPI_Cart_coords(comm_cart, procID, 2, start);
      start[0] *= nxnom;
      start[1] *= nynom;
      MPI_Type_create_subarray(2, size, subsize, start, MPI_ORDER_C, MPI_INT, &file_type);
      MPI_Type_commit(&file_type);
      size[0] = nx;
      size[1] = ny;
      subsize[0] = nxnom;
      subsize[1] = nynom;
      start[0] = 1;
      start[1] = 1;
      MPI_Type_create_subarray(2, size, subsize, start, MPI_ORDER_C, MPI_INT, &mem_type);
      MPI_Type_commit(&mem_type);

      /* open file for MPI-2 parallel i/o */
      MPI_File_open(MPI_COMM_WORLD, "wavetoy_timelevel.dat", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
      MPI_File_set_view(fh, 0, MPI_INT, file_type, "native", MPI_INFO_NULL);
      MPI_File_write_all(fh, &unew[0][0], 1, mem_type, &status);
      MPI_File_close(&fh);
   }

   printf("Process %i of %i finished.\n", procID, nprocs);
   MPI_Finalize();
}
