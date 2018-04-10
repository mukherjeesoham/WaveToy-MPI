/*============================================================*/
/* Solve a 2D scalar wave equation                            */
/* MPI communication implementation                           */
/* Generalize to n processes in each direction                */
/* Soham 3/2018                                               */
/* FIXME: For no. of points > 1 in each process, sometimes the 
 * code fails                                                 */
/*============================================================*/

#include<mpi.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

#define nx 4      // no. of points in x
#define ny 4      // no. of points in y

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
    int proc_north, proc_south, proc_east, proc_west,
        proc_se, proc_sw, proc_nw, proc_ne;
    MPI_Status status;
    MPI_Datatype row_type, column_type;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);

    // number of processors in each direction
    nxproc = 4;
    nyproc = 4;

    // Fix number of request handles you need
    MPI_Request request[16*nxproc*nyproc];
    
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

    // Define MPI datatypes for contiguous and non-contiguous data
    MPI_Type_contiguous(nynom, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);
    MPI_Type_vector(nxnom, 1, gny, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    // Create a cart-grid
    cartID[1] = procID%nxproc;
    cartID[0] = (procID/nxproc)%nyproc;

    // Figure out surroudings
    proc_north = sub2ind(cartID[0] - 1, cartID[1],     nxproc, nyproc);
    proc_south = sub2ind(cartID[0] + 1, cartID[1],     nxproc, nyproc);
    proc_east  = sub2ind(cartID[0],     cartID[1] + 1, nxproc, nyproc);
    proc_west  = sub2ind(cartID[0],     cartID[1] - 1, nxproc, nyproc);
    proc_nw    = sub2ind(cartID[0] - 1, cartID[1] - 1, nxproc, nyproc);
    proc_ne    = sub2ind(cartID[0] - 1, cartID[1] + 1, nxproc, nyproc);
    proc_se    = sub2ind(cartID[0] + 1, cartID[1] + 1, nxproc, nyproc);
    proc_sw    = sub2ind(cartID[0] + 1, cartID[1] - 1, nxproc, nyproc);

    // Initiate communication: chooses processes and how they act
    if (cartID[0] == 0 && cartID[1] == 0){
          MPI_Isend(&uold[1][iyem],    1, column_type, proc_east,   0, MPI_COMM_WORLD, &request[(procID*16)+0]);  // exchange RC
          MPI_Irecv(&uold[1][iye],     1, column_type, proc_east,   0, MPI_COMM_WORLD, &request[(procID*16)+1]);
          MPI_Isend(&uold[ixem][1],    1, row_type,    proc_south,  0, MPI_COMM_WORLD, &request[(procID*16)+2]);  // exchange BR
          MPI_Irecv(&uold[ixe][1],     1, row_type,    proc_south,  0, MPI_COMM_WORLD, &request[(procID*16)+3]);
          MPI_Isend(&uold[ixem][iyem], 1, MPI_DOUBLE,  proc_se,     0, MPI_COMM_WORLD, &request[(procID*16)+4]);  // exchange SEC
          MPI_Irecv(&uold[ixe][iye],   1, MPI_DOUBLE,  proc_se,     0, MPI_COMM_WORLD, &request[(procID*16)+5]);

          for (i=procID*16; i<=(procID*16)+5; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartID[0] == 0 && cartID[1] == (nyproc-1)){
          MPI_Isend(&uold[1][1],       1, column_type, proc_west,   0, MPI_COMM_WORLD, &request[(procID*16)+0]);  // exchange LC
          MPI_Irecv(&uold[1][0],       1, column_type, proc_west,   0, MPI_COMM_WORLD, &request[(procID*16)+1]);
          MPI_Isend(&uold[ixem][1],    1, row_type,    proc_south,  0, MPI_COMM_WORLD, &request[(procID*16)+2]);  // exchange BR
          MPI_Irecv(&uold[ixe][1],     1, row_type,    proc_south,  0, MPI_COMM_WORLD, &request[(procID*16)+3]);
          MPI_Isend(&uold[ixem][1],    1, MPI_DOUBLE,  proc_sw,     0, MPI_COMM_WORLD, &request[(procID*16)+4]); // exchange SWC
          MPI_Irecv(&uold[ixe][0],     1, MPI_DOUBLE,  proc_sw,     0, MPI_COMM_WORLD, &request[(procID*16)+5]);

          for (i=procID*16; i<=(procID*16)+5; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartID[0] == (nxproc-1) && cartID[1] == 0) {
          MPI_Isend(&uold[1][iyem],    1, column_type, proc_east,   0, MPI_COMM_WORLD, &request[(procID*16)+0]);  // exchange RC
          MPI_Irecv(&uold[1][iye],     1, column_type, proc_east,   0, MPI_COMM_WORLD, &request[(procID*16)+1]);
          MPI_Isend(&uold[1][1],       1, row_type,    proc_north,  0, MPI_COMM_WORLD, &request[(procID*16)+2]);  // exchange TR
          MPI_Irecv(&uold[0][1],       1, row_type,    proc_north,  0, MPI_COMM_WORLD, &request[(procID*16)+3]);
          MPI_Isend(&uold[1][iyem],    1, MPI_DOUBLE,  proc_ne,     0, MPI_COMM_WORLD, &request[(procID*16)+4]); // exchange NEC
          MPI_Irecv(&uold[0][iye],     1, MPI_DOUBLE,  proc_ne,     0, MPI_COMM_WORLD, &request[(procID*16)+5]);

          for (i=procID*16; i<=(procID*16)+5; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartID[0] == (nxproc-1) && cartID[1] == (nyproc-1)) {
          MPI_Isend(&uold[1][1],       1, column_type, proc_west,   0, MPI_COMM_WORLD, &request[(procID*16)+0]);  // exchange LC
          MPI_Irecv(&uold[1][0],       1, column_type, proc_west,   0, MPI_COMM_WORLD, &request[(procID*16)+1]);
          MPI_Isend(&uold[1][1],       1, row_type,    proc_north,  0, MPI_COMM_WORLD, &request[(procID*16)+2]);  // exchange TR
          MPI_Irecv(&uold[0][1],       1, row_type,    proc_north,  0, MPI_COMM_WORLD, &request[(procID*16)+3]);
          MPI_Isend(&uold[1][1],       1, MPI_DOUBLE,  proc_nw,     0, MPI_COMM_WORLD, &request[(procID*16)+4]);  // exchange NWC
          MPI_Irecv(&uold[0][0],       1, MPI_DOUBLE,  proc_nw,     0, MPI_COMM_WORLD, &request[(procID*16)+5]);

          for (i=procID*16; i<=(procID*16)+5; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartID[0] == 0) {
          MPI_Isend(&uold[1][1],       1, column_type, proc_west,   0, MPI_COMM_WORLD, &request[(procID*16)+0]);  // exchange LC
          MPI_Irecv(&uold[1][0],       1, column_type, proc_west,   0, MPI_COMM_WORLD, &request[(procID*16)+1]);
          MPI_Isend(&uold[1][iyem],    1, column_type, proc_east,   0, MPI_COMM_WORLD, &request[(procID*16)+2]);  // exchange RC
          MPI_Irecv(&uold[1][iye],     1, column_type, proc_east,   0, MPI_COMM_WORLD, &request[(procID*16)+3]);
          MPI_Isend(&uold[ixem][1],    1, row_type,    proc_south,  0, MPI_COMM_WORLD, &request[(procID*16)+4]);  // exchange BR
          MPI_Irecv(&uold[ixe][1],     1, row_type,    proc_south,  0, MPI_COMM_WORLD, &request[(procID*16)+5]);
          MPI_Isend(&uold[ixem][1],    1, MPI_DOUBLE,  proc_sw,     0, MPI_COMM_WORLD, &request[(procID*16)+6]);  // exchange SWC
          MPI_Irecv(&uold[ixe][0],     1, MPI_DOUBLE,  proc_sw,     0, MPI_COMM_WORLD, &request[(procID*16)+7]);
          MPI_Isend(&uold[ixem][iyem], 1, MPI_DOUBLE,  proc_se,     0, MPI_COMM_WORLD, &request[(procID*16)+8]);  // exchange SEC
          MPI_Irecv(&uold[ixe][iye],   1, MPI_DOUBLE,  proc_se,     0, MPI_COMM_WORLD, &request[(procID*16)+9]);

          for (i=procID*16; i<=(procID*16)+9; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartID[0] == (nxproc-1)) {
          MPI_Isend(&uold[1][1],       1, column_type, proc_west,   0, MPI_COMM_WORLD, &request[(procID*16)+0]);  // exchange LC
          MPI_Irecv(&uold[1][0],       1, column_type, proc_west,   0, MPI_COMM_WORLD, &request[(procID*16)+1]);
          MPI_Isend(&uold[1][iyem],    1, column_type, proc_east,   0, MPI_COMM_WORLD, &request[(procID*16)+2]);  // exchange RC
          MPI_Irecv(&uold[1][iye],     1, column_type, proc_east,   0, MPI_COMM_WORLD, &request[(procID*16)+3]);
          MPI_Isend(&uold[1][1],       1, row_type,    proc_north,  0, MPI_COMM_WORLD, &request[(procID*16)+4]);  // exchange TR
          MPI_Irecv(&uold[0][1],       1, row_type,    proc_north,  0, MPI_COMM_WORLD, &request[(procID*16)+5]);
          MPI_Isend(&uold[1][1],       1, MPI_DOUBLE,  proc_nw,     0, MPI_COMM_WORLD, &request[(procID*16)+6]);  // exchange NWC
          MPI_Irecv(&uold[0][0],       1, MPI_DOUBLE,  proc_nw,     0, MPI_COMM_WORLD, &request[(procID*16)+7]);
          MPI_Isend(&uold[1][iyem],    1, MPI_DOUBLE,  proc_ne,     0, MPI_COMM_WORLD, &request[(procID*16)+8]);  // exchange NEC
          MPI_Irecv(&uold[0][iye],     1, MPI_DOUBLE,  proc_ne,     0, MPI_COMM_WORLD, &request[(procID*16)+9]);

          for (i=procID*16; i<=(procID*16)+9; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartID[1] == 0) {
          MPI_Isend(&uold[1][iyem],    1, column_type, proc_east,   0, MPI_COMM_WORLD, &request[(procID*16)+0]);  // exchange RC
          MPI_Irecv(&uold[1][iye],     1, column_type, proc_east,   0, MPI_COMM_WORLD, &request[(procID*16)+1]);
          MPI_Isend(&uold[1][1],       1, row_type,    proc_north,  0, MPI_COMM_WORLD, &request[(procID*16)+2]);  // exchange TR
          MPI_Irecv(&uold[0][1],       1, row_type,    proc_north,  0, MPI_COMM_WORLD, &request[(procID*16)+3]);
          MPI_Isend(&uold[ixem][1],    1, row_type,    proc_south,  0, MPI_COMM_WORLD, &request[(procID*16)+4]);  // exchange BR
          MPI_Irecv(&uold[ixe][1],     1, row_type,    proc_south,  0, MPI_COMM_WORLD, &request[(procID*16)+5]);
          MPI_Isend(&uold[1][iyem],    1, MPI_DOUBLE,  proc_ne,     0, MPI_COMM_WORLD, &request[(procID*16)+6]);  // exchange NEC
          MPI_Irecv(&uold[0][iye],     1, MPI_DOUBLE,  proc_ne,     0, MPI_COMM_WORLD, &request[(procID*16)+7]);
          MPI_Isend(&uold[ixem][iyem], 1, MPI_DOUBLE,  proc_se,     0, MPI_COMM_WORLD, &request[(procID*16)+8]);  // exchange SEC
          MPI_Irecv(&uold[ixe][iye],   1, MPI_DOUBLE,  proc_se,     0, MPI_COMM_WORLD, &request[(procID*16)+9]);

          for (i=procID*16; i<=(procID*16)+9; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartID[1] == (nyproc-1)) {
          MPI_Isend(&uold[1][1],       1, column_type, proc_west,   0, MPI_COMM_WORLD, &request[(procID*16)+0]);  // exchange LC
          MPI_Irecv(&uold[1][0],       1, column_type, proc_west,   0, MPI_COMM_WORLD, &request[(procID*16)+1]);
          MPI_Isend(&uold[1][1],       1, row_type,    proc_north,  0, MPI_COMM_WORLD, &request[(procID*16)+2]);  // exchange TR
          MPI_Irecv(&uold[0][1],       1, row_type,    proc_north,  0, MPI_COMM_WORLD, &request[(procID*16)+3]);
          MPI_Isend(&uold[ixem][1],    1, row_type,    proc_south,  0, MPI_COMM_WORLD, &request[(procID*16)+4]);  // exchange BR
          MPI_Irecv(&uold[ixe][1],     1, row_type,    proc_south,  0, MPI_COMM_WORLD, &request[(procID*16)+5]);
          MPI_Isend(&uold[1][1],       1, MPI_DOUBLE,  proc_nw,     0, MPI_COMM_WORLD, &request[(procID*16)+6]);  // exchange NWC
          MPI_Irecv(&uold[0][0],       1, MPI_DOUBLE,  proc_nw,     0, MPI_COMM_WORLD, &request[(procID*16)+7]);
          MPI_Isend(&uold[ixem][1],    1, MPI_DOUBLE,  proc_sw,     0, MPI_COMM_WORLD, &request[(procID*16)+8]);  // exchange SWC
          MPI_Irecv(&uold[ixe][0],     1, MPI_DOUBLE,  proc_sw,     0, MPI_COMM_WORLD, &request[(procID*16)+9]);

          for (i=procID*16; i<=(procID*16)+9; i++){
              MPI_Wait(&request[i], &status);
          }
    } else {
        MPI_Isend(&uold[1][1],        1, column_type, proc_west,   0, MPI_COMM_WORLD, &request[(procID*16)+0]);   // exchange LC
        MPI_Irecv(&uold[1][0],        1, column_type, proc_west,   0, MPI_COMM_WORLD, &request[(procID*16)+1]);
        MPI_Isend(&uold[1][iyem],     1, column_type, proc_east,   0, MPI_COMM_WORLD, &request[(procID*16)+2]);   // exchange RC
        MPI_Irecv(&uold[1][iye],      1, column_type, proc_east,   0, MPI_COMM_WORLD, &request[(procID*16)+3]);
        MPI_Isend(&uold[1][1],        1, row_type,    proc_north,  0, MPI_COMM_WORLD, &request[(procID*16)+4]);   // exchange TR
        MPI_Irecv(&uold[0][1],        1, row_type,    proc_north,  0, MPI_COMM_WORLD, &request[(procID*16)+5]);
        MPI_Isend(&uold[ixem][1],     1, row_type,    proc_south,  0, MPI_COMM_WORLD, &request[(procID*16)+6]);   // exchange BR
        MPI_Irecv(&uold[ixe][1],      1, row_type,    proc_south,  0, MPI_COMM_WORLD, &request[(procID*16)+7]);
        MPI_Isend(&uold[1][1],        1, MPI_DOUBLE,  proc_nw,     0, MPI_COMM_WORLD, &request[(procID*16)+8]);   // exchange NWC
        MPI_Irecv(&uold[0][0],        1, MPI_DOUBLE,  proc_nw,     0, MPI_COMM_WORLD, &request[(procID*16)+9]);
        MPI_Isend(&uold[1][iyem],     1, MPI_DOUBLE,  proc_ne,     0, MPI_COMM_WORLD, &request[(procID*16)+10]);  // exchange NEC
        MPI_Irecv(&uold[0][iye],      1, MPI_DOUBLE,  proc_ne,     0, MPI_COMM_WORLD, &request[(procID*16)+11]);
        MPI_Isend(&uold[ixem][1],     1, MPI_DOUBLE,  proc_sw,     0, MPI_COMM_WORLD, &request[(procID*16)+12]);  // exchange SWC
        MPI_Irecv(&uold[ixe][0],      1, MPI_DOUBLE,  proc_sw,     0, MPI_COMM_WORLD, &request[(procID*16)+13]);
        MPI_Isend(&uold[ixem][iyem],  1, MPI_DOUBLE,  proc_se,     0, MPI_COMM_WORLD, &request[(procID*16)+14]);  // exchange SEC
        MPI_Irecv(&uold[ixe][iye],    1, MPI_DOUBLE,  proc_se,     0, MPI_COMM_WORLD, &request[(procID*16)+15]);

        for (i=procID*16; i<=(procID*16)+15; i++){
            MPI_Wait(&request[i], &status);
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

    MPI_Finalize();
}
