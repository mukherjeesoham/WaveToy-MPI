/*============================================================*/
/* Test MPI_Reduce operation                                  */
/* Soham 3/2018                                               */
/*============================================================*/

#include<mpi.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

int sub2ind(int ix, int iy, int nxprocs){
    return iy + nxprocs*ix;
}

int main(int argc, char *argv[]){
    int i, j, li, lj;
    int gixs, gixe, giys, giye;
    int lixs, lixe, liys, liye, lixm, liym;
    int nx, ny, tnx, tny, tnxtny;
    int nxprocs, nyprocs, nxnom, nynom, nprocs, rank;
    int cartrank[2];
    double **uold, *uold1d;
    double x, y, sum, sumreduce;
    int proc_north, proc_south, proc_east, proc_west,
        proc_se, proc_sw, proc_nw, proc_ne;
    MPI_Status status;
    MPI_Datatype rtype, ctype;

    // Number of points in each direction (without ghost zones)
    sscanf(argv[1], "%i", &nx);      
    sscanf(argv[2], "%i", &ny);      
   
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request request[16*nprocs];
    
    // number of processors in each direction [Fair division?]
    nxprocs = (nprocs*nx)/(nx+ny);
    nyprocs = (nprocs*ny)/(nx+ny);

    // Abort if work can't be divided nicely.
    if (nxprocs == 0 || nyprocs == 0 || nxnom == 0 || nynom == 0 || nprocs < 4) {
        if (rank == 0) printf("ERROR: Could not (nicely) divide the work among total number of processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // nominal number of points in each patch without ghost zones
    nxnom = nx/nxprocs;
    nynom = ny/nyprocs;
    
    // Print some info to screen
    if (rank == 0){
        printf("------------------------------------------------------------------------\n");
        printf("Starting Wavetoy-MPI\n");
        printf("------------------------------------------------------------------------\n");
        printf("-- Basic MPI Info\n");
        printf("   | Number of points along x = %i\n", nx);
        printf("   | Number of points along y = %i\n", ny);
        printf("   | Number of procs  along x = %i\n", nxprocs);
        printf("   | Number of procs  along y = %i\n", nyprocs);
    }
    
    // global indices without ghost zones ('s' and 'e' refer to start and end)
    gixs = ((rank/nxprocs)*nxnom) + 1;  
    gixe = gixs + nxnom - 1;             
    giys = ((rank%nyprocs)*nynom) + 1;  
    giye = giys + nynom - 1;             
    
    // local starting and ending indices (including ghost zones)
    lixs = 0;
    lixe = nxnom + 1;
    liys = 0;
    liye = nynom + 1;

    // ending index minus 1 (useful for exchanging ghost zones)
    lixm = lixe - 1;
    liym = liye - 1;

    // Define (total number of points) array sizes including ghost zones (or nxnom + 2)
    tnx = nxnom + 2;
    tny = nynom + 2;
    tnxtny = tnx*tny;

    // Allocate memory for arrays
    uold1d = malloc(tnxtny*sizeof(double));
    uold   = malloc(tnx*sizeof(double*));  

    for(i=0; i<tnx; i++){      
      uold[i] = &uold1d[i*tny];  
    }
    
    // Initialize uold
    for (i=0; i<=lixe; i++){
        for (j=0; j<=liye; j++){
            if (i == 0 || j == 0 || i == lixe || j == liye){
                uold[i][j] = 0.0;
            } else {
                uold[i][j] = (double)rank + 1.0;
            }
        }
    }

    // Define MPI datatypes for contiguous and non-contiguous data
    MPI_Type_contiguous(nynom, MPI_DOUBLE, &rtype);
    MPI_Type_commit(&rtype);
    MPI_Type_vector(nxnom, 1, tny, MPI_DOUBLE, &ctype);
    MPI_Type_commit(&ctype);

    // Create a cart-grid
    cartrank[1] = rank%nxprocs;
    cartrank[0] = (rank/nxprocs)%nyprocs;

    // Figure out surroudings
    proc_north = sub2ind(cartrank[0] - 1, cartrank[1],     nxprocs);
    proc_south = sub2ind(cartrank[0] + 1, cartrank[1],     nxprocs);
    proc_east  = sub2ind(cartrank[0],     cartrank[1] + 1, nxprocs);
    proc_west  = sub2ind(cartrank[0],     cartrank[1] - 1, nxprocs);
    proc_nw    = sub2ind(cartrank[0] - 1, cartrank[1] - 1, nxprocs);
    proc_ne    = sub2ind(cartrank[0] - 1, cartrank[1] + 1, nxprocs);
    proc_se    = sub2ind(cartrank[0] + 1, cartrank[1] + 1, nxprocs);
    proc_sw    = sub2ind(cartrank[0] + 1, cartrank[1] - 1, nxprocs);

    // Initiate communication: chooses processes and how they act
    if (cartrank[0] == 0 && cartrank[1] == 0){  // CTL
          MPI_Isend(&uold[1][liym],    1, ctype,    proc_east, 0, MPI_COMM_WORLD, &request[(rank*16)+0]);  // exchange RC
          MPI_Irecv(&uold[1][liye],    1, ctype,    proc_east, 0, MPI_COMM_WORLD, &request[(rank*16)+1]);
          MPI_Isend(&uold[lixm][1],    1, rtype,   proc_south, 0, MPI_COMM_WORLD, &request[(rank*16)+2]);  // exchange BR
          MPI_Irecv(&uold[lixe][1],    1, rtype,   proc_south, 0, MPI_COMM_WORLD, &request[(rank*16)+3]);
          MPI_Isend(&uold[lixm][liym], 1, MPI_DOUBLE, proc_se, 0, MPI_COMM_WORLD, &request[(rank*16)+4]);  // exchange SEC
          MPI_Irecv(&uold[lixe][liye], 1, MPI_DOUBLE, proc_se, 0, MPI_COMM_WORLD, &request[(rank*16)+5]);

          for (i=rank*16; i<=(rank*16)+5; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartrank[0] == 0 && cartrank[1] == (nyprocs-1)){ // CBL
          MPI_Isend(&uold[1][1],      1, ctype,    proc_west, 0, MPI_COMM_WORLD, &request[(rank*16)+0]);  // exchange LC
          MPI_Irecv(&uold[1][0],      1, ctype,    proc_west, 0, MPI_COMM_WORLD, &request[(rank*16)+1]);
          MPI_Isend(&uold[lixm][1],   1, rtype,   proc_south, 0, MPI_COMM_WORLD, &request[(rank*16)+2]);  // exchange BR
          MPI_Irecv(&uold[lixe][1],   1, rtype,   proc_south, 0, MPI_COMM_WORLD, &request[(rank*16)+3]);
          MPI_Isend(&uold[lixm][1],   1, MPI_DOUBLE, proc_sw, 0, MPI_COMM_WORLD, &request[(rank*16)+4]);  // exchange SWC
          MPI_Irecv(&uold[lixe][0],   1, MPI_DOUBLE, proc_sw, 0, MPI_COMM_WORLD, &request[(rank*16)+5]);

          for (i=rank*16; i<=(rank*16)+5; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartrank[0] == (nxprocs-1) && cartrank[1] == 0) {    // CTR
          MPI_Isend(&uold[1][liym], 1, ctype,    proc_east, 0, MPI_COMM_WORLD, &request[(rank*16)+0]);  // exchange RC
          MPI_Irecv(&uold[1][liye], 1, ctype,    proc_east, 0, MPI_COMM_WORLD, &request[(rank*16)+1]);
          MPI_Isend(&uold[1][1],    1, rtype,   proc_north, 0, MPI_COMM_WORLD, &request[(rank*16)+2]);  // exchange TR
          MPI_Irecv(&uold[0][1],    1, rtype,   proc_north, 0, MPI_COMM_WORLD, &request[(rank*16)+3]);
          MPI_Isend(&uold[1][liym], 1, MPI_DOUBLE, proc_ne, 0, MPI_COMM_WORLD, &request[(rank*16)+4]);  // exchange NEC
          MPI_Irecv(&uold[0][liye], 1, MPI_DOUBLE, proc_ne, 0, MPI_COMM_WORLD, &request[(rank*16)+5]);

          for (i=rank*16; i<=(rank*16)+5; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartrank[0] == (nxprocs-1) && cartrank[1] == (nyprocs-1)) {  // CBR
          MPI_Isend(&uold[1][1], 1, ctype,    proc_west, 0, MPI_COMM_WORLD, &request[(rank*16)+0]);  // exchange LC
          MPI_Irecv(&uold[1][0], 1, ctype,    proc_west, 0, MPI_COMM_WORLD, &request[(rank*16)+1]);
          MPI_Isend(&uold[1][1], 1, rtype,   proc_north, 0, MPI_COMM_WORLD, &request[(rank*16)+2]);  // exchange TR
          MPI_Irecv(&uold[0][1], 1, rtype,   proc_north, 0, MPI_COMM_WORLD, &request[(rank*16)+3]);
          MPI_Isend(&uold[1][1], 1, MPI_DOUBLE, proc_nw, 0, MPI_COMM_WORLD, &request[(rank*16)+4]);  // exchange NWC
          MPI_Irecv(&uold[0][0], 1, MPI_DOUBLE, proc_nw, 0, MPI_COMM_WORLD, &request[(rank*16)+5]);

          for (i=rank*16; i<=(rank*16)+5; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartrank[0] == 0) {  // TR
          MPI_Isend(&uold[1][1],       1, ctype,    proc_west, 0, MPI_COMM_WORLD, &request[(rank*16)+0]);  // exchange LC
          MPI_Irecv(&uold[1][0],       1, ctype,    proc_west, 0, MPI_COMM_WORLD, &request[(rank*16)+1]);
          MPI_Isend(&uold[1][liym],    1, ctype,    proc_east, 0, MPI_COMM_WORLD, &request[(rank*16)+2]);  // exchange RC
          MPI_Irecv(&uold[1][liye],    1, ctype,    proc_east, 0, MPI_COMM_WORLD, &request[(rank*16)+3]);
          MPI_Isend(&uold[lixm][1],    1, rtype,   proc_south, 0, MPI_COMM_WORLD, &request[(rank*16)+4]);  // exchange BR
          MPI_Irecv(&uold[lixe][1],    1, rtype,   proc_south, 0, MPI_COMM_WORLD, &request[(rank*16)+5]);
          MPI_Isend(&uold[lixm][1],    1, MPI_DOUBLE, proc_sw, 0, MPI_COMM_WORLD, &request[(rank*16)+6]);  // exchange SWC
          MPI_Irecv(&uold[lixe][0],    1, MPI_DOUBLE, proc_sw, 0, MPI_COMM_WORLD, &request[(rank*16)+7]);
          MPI_Isend(&uold[lixm][liym], 1, MPI_DOUBLE, proc_se, 0, MPI_COMM_WORLD, &request[(rank*16)+8]);  // exchange SEC
          MPI_Irecv(&uold[lixe][liye], 1, MPI_DOUBLE, proc_se, 0, MPI_COMM_WORLD, &request[(rank*16)+9]);

          for (i=rank*16; i<=(rank*16)+9; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartrank[0] == (nxprocs-1)) {    // BR
          MPI_Isend(&uold[1][1],    1, ctype,    proc_west, 0, MPI_COMM_WORLD, &request[(rank*16)+0]);  // exchange LC
          MPI_Irecv(&uold[1][0],    1, ctype,    proc_west, 0, MPI_COMM_WORLD, &request[(rank*16)+1]);
          MPI_Isend(&uold[1][liym], 1, ctype,    proc_east, 0, MPI_COMM_WORLD, &request[(rank*16)+2]);  // exchange RC
          MPI_Irecv(&uold[1][liye], 1, ctype,    proc_east, 0, MPI_COMM_WORLD, &request[(rank*16)+3]);
          MPI_Isend(&uold[1][1],    1, rtype,   proc_north, 0, MPI_COMM_WORLD, &request[(rank*16)+4]);  // exchange TR
          MPI_Irecv(&uold[0][1],    1, rtype,   proc_north, 0, MPI_COMM_WORLD, &request[(rank*16)+5]);
          MPI_Isend(&uold[1][1],    1, MPI_DOUBLE, proc_nw, 0, MPI_COMM_WORLD, &request[(rank*16)+6]);  // exchange NWC
          MPI_Irecv(&uold[0][0],    1, MPI_DOUBLE, proc_nw, 0, MPI_COMM_WORLD, &request[(rank*16)+7]);
          MPI_Isend(&uold[1][liym], 1, MPI_DOUBLE, proc_ne, 0, MPI_COMM_WORLD, &request[(rank*16)+8]);  // exchange NEC
          MPI_Irecv(&uold[0][liye], 1, MPI_DOUBLE, proc_ne, 0, MPI_COMM_WORLD, &request[(rank*16)+9]);

          for (i=rank*16; i<=(rank*16)+9; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartrank[1] == 0) {  // LC
          MPI_Isend(&uold[1][liym],    1, ctype,    proc_east, 0, MPI_COMM_WORLD, &request[(rank*16)+0]);  // exchange RC
          MPI_Irecv(&uold[1][liye],    1, ctype,    proc_east, 0, MPI_COMM_WORLD, &request[(rank*16)+1]);
          MPI_Isend(&uold[1][1],       1, rtype,   proc_north, 0, MPI_COMM_WORLD, &request[(rank*16)+2]);  // exchange TR
          MPI_Irecv(&uold[0][1],       1, rtype,   proc_north, 0, MPI_COMM_WORLD, &request[(rank*16)+3]);
          MPI_Isend(&uold[lixm][1],    1, rtype,   proc_south, 0, MPI_COMM_WORLD, &request[(rank*16)+4]);  // exchange BR
          MPI_Irecv(&uold[lixe][1],    1, rtype,   proc_south, 0, MPI_COMM_WORLD, &request[(rank*16)+5]);
          MPI_Isend(&uold[1][liym],    1, MPI_DOUBLE, proc_ne, 0, MPI_COMM_WORLD, &request[(rank*16)+6]);  // exchange NEC
          MPI_Irecv(&uold[0][liye],    1, MPI_DOUBLE, proc_ne, 0, MPI_COMM_WORLD, &request[(rank*16)+7]);
          MPI_Isend(&uold[lixm][liym], 1, MPI_DOUBLE, proc_se, 0, MPI_COMM_WORLD, &request[(rank*16)+8]);  // exchange SEC
          MPI_Irecv(&uold[lixe][liye], 1, MPI_DOUBLE, proc_se, 0, MPI_COMM_WORLD, &request[(rank*16)+9]);

          for (i=rank*16; i<=(rank*16)+9; i++){
              MPI_Wait(&request[i], &status);
          }
    } else if (cartrank[1] == (nyprocs-1)) {    // RC
          MPI_Isend(&uold[1][1],    1, ctype,    proc_west, 0, MPI_COMM_WORLD, &request[(rank*16)+0]);  // exchange LC
          MPI_Irecv(&uold[1][0],    1, ctype,    proc_west, 0, MPI_COMM_WORLD, &request[(rank*16)+1]);
          MPI_Isend(&uold[1][1],    1, rtype,   proc_north, 0, MPI_COMM_WORLD, &request[(rank*16)+2]);  // exchange TR
          MPI_Irecv(&uold[0][1],    1, rtype,   proc_north, 0, MPI_COMM_WORLD, &request[(rank*16)+3]);
          MPI_Isend(&uold[lixm][1], 1, rtype,   proc_south, 0, MPI_COMM_WORLD, &request[(rank*16)+4]);  // exchange BR
          MPI_Irecv(&uold[lixe][1], 1, rtype,   proc_south, 0, MPI_COMM_WORLD, &request[(rank*16)+5]);
          MPI_Isend(&uold[1][1],    1, MPI_DOUBLE, proc_nw, 0, MPI_COMM_WORLD, &request[(rank*16)+6]);  // exchange NWC
          MPI_Irecv(&uold[0][0],    1, MPI_DOUBLE, proc_nw, 0, MPI_COMM_WORLD, &request[(rank*16)+7]);
          MPI_Isend(&uold[lixm][1], 1, MPI_DOUBLE, proc_sw, 0, MPI_COMM_WORLD, &request[(rank*16)+8]);  // exchange SWC
          MPI_Irecv(&uold[lixe][0], 1, MPI_DOUBLE, proc_sw, 0, MPI_COMM_WORLD, &request[(rank*16)+9]);

          for (i=rank*16; i<=(rank*16)+9; i++){
              MPI_Wait(&request[i], &status);
          }
    } else {    // BTW
        MPI_Isend(&uold[1][1],       1, ctype,    proc_west, 0, MPI_COMM_WORLD, &request[(rank*16)+0]);   // exchange LC
        MPI_Irecv(&uold[1][0],       1, ctype,    proc_west, 0, MPI_COMM_WORLD, &request[(rank*16)+1]);
        MPI_Isend(&uold[1][liym],    1, ctype,    proc_east, 0, MPI_COMM_WORLD, &request[(rank*16)+2]);   // exchange RC
        MPI_Irecv(&uold[1][liye],    1, ctype,    proc_east, 0, MPI_COMM_WORLD, &request[(rank*16)+3]);
        MPI_Isend(&uold[1][1],       1, rtype,   proc_north, 0, MPI_COMM_WORLD, &request[(rank*16)+4]);   // exchange TR
        MPI_Irecv(&uold[0][1],       1, rtype,   proc_north, 0, MPI_COMM_WORLD, &request[(rank*16)+5]);
        MPI_Isend(&uold[lixm][1],    1, rtype,   proc_south, 0, MPI_COMM_WORLD, &request[(rank*16)+6]);   // exchange BR
        MPI_Irecv(&uold[lixe][1],    1, rtype,   proc_south, 0, MPI_COMM_WORLD, &request[(rank*16)+7]);
        MPI_Isend(&uold[1][1],       1, MPI_DOUBLE, proc_nw, 0, MPI_COMM_WORLD, &request[(rank*16)+8]);   // exchange NWC
        MPI_Irecv(&uold[0][0],       1, MPI_DOUBLE, proc_nw, 0, MPI_COMM_WORLD, &request[(rank*16)+9]);
        MPI_Isend(&uold[1][liym],    1, MPI_DOUBLE, proc_ne, 0, MPI_COMM_WORLD, &request[(rank*16)+10]);  // exchange NEC
        MPI_Irecv(&uold[0][liye],    1, MPI_DOUBLE, proc_ne, 0, MPI_COMM_WORLD, &request[(rank*16)+11]);
        MPI_Isend(&uold[lixm][1],    1, MPI_DOUBLE, proc_sw, 0, MPI_COMM_WORLD, &request[(rank*16)+12]);  // exchange SWC
        MPI_Irecv(&uold[lixe][0],    1, MPI_DOUBLE, proc_sw, 0, MPI_COMM_WORLD, &request[(rank*16)+13]);
        MPI_Isend(&uold[lixm][liym], 1, MPI_DOUBLE, proc_se, 0, MPI_COMM_WORLD, &request[(rank*16)+14]);  // exchange SEC
        MPI_Irecv(&uold[lixe][liye], 1, MPI_DOUBLE, proc_se, 0, MPI_COMM_WORLD, &request[(rank*16)+15]);

        for (i=rank*16; i<=(rank*16)+15; i++){
            MPI_Wait(&request[i], &status);
        }
    }
    
    if(1){
        printf("------------------------------------\n");
        printf("proc[%i] \n", rank);
        printf("------------------------------------\n");
        for (i=0; i<=lixe; i++){
            for (j=0; j<=liye; j++){
                printf("%1.1f\t", uold[i][j]);
            }
            printf("\n");
        }
    }
  
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) printf("-- All done. Exiting MPI environment.\n");
    
    MPI_Finalize();
}
