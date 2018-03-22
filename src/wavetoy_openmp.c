/*============================================================*/
/* Solve a 2D scalar wave equation                            */
/* Added OpenMP for evolution loop                            */
/* Soham 3/2018                                               */
/*============================================================*/

#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define nx 100      // no. of points in x
#define ny 100      // no. of points in y
#define nsteps 10   // no. of time steps

int main(){
    int **uold, **unew, **ucur;
    int gnx, gny;
    int i, j, k;
    double c, csq, x, y, dx, dy, dt, dtdx, dtdxsq;
    
    // Increase the array size to include ghost zones
    gnx = nx + 2;
    gny = ny + 2;

    // Some basic quantities 
    c  = 1.0;                       // wave speed
    dt = 1.0/(double)(nsteps-1);    // final time is 1
    dx = 2.0/(double)(nx+1);        // x spans from -1 to 1
    dy = 2.0/(double)(ny+1);        // y spans from -1 to 1
    csq = c*c;  
    dtdx = dt/dx;
    dtdxsq = dtdx*dtdx;
    
    // Allocate memory dynamically for these arrays
    uold = malloc(gnx*sizeof(double*));
    ucur = malloc(gnx*sizeof(double*));
    unew = malloc(gnx*sizeof(double*));

    for (i=0; i<=gny; i++){
        uold[i] = malloc(gny*sizeof(double*));
        ucur[i] = malloc(gny*sizeof(double*));
        unew[i] = malloc(gny*sizeof(double*));
    }

    // Initialize uold
    for (i=1; i<=nx; i++){
        for (j=1; j<=ny; j++){
            x = -(double)(1 + nx + 2*i)/(double)(1 + nx); 
            y = -(double)(1 + ny + 2*j)/(double)(1 + ny);
            uold[i][j] = exp(-x*x/0.01 - y*y/0.01);        // IC: Gaussian
        }
    }

    // Initialize ucur
    for (i=1; i<=nx; i++){
        for (j=1; j<=ny; j++){
            x = -(double)(1 + nx + 2*i)/(double)(1 + nx); 
            y = -(double)(1 + ny + 2*j)/(double)(1 + ny);
            ucur[i][j] = uold[i][j] - 2.0*dt*0.0           // IC: moment of time symmetry dudt = 0 
                         + 0.5*csq*dtdxsq*(uold[i+1][j] - 2.0*uold[i][j] + uold[i-1][j]
                                         + uold[i][j+1] - 2.0*uold[i][j] + uold[i][j-1]);    
        }
    }

    printf("Jetzt geht's los!\n");
    printf("Running on %i threads\n", omp_get_num_threads());    
    
    // update solution to next step for n steps
    #pragma omp parallel for
    for (k=0; k<=nsteps; k++){
        // Setting up boundary conditions. 
        // This can be done differently, but while exchaning data, an explicit 
        // set up would be useful. 
        
        // Impose Dirichlet BCs on the corners
        uold[0][0]    = 0.0;
        uold[0][0]    = 0.0;
        ucur[nx+1][0] = 0.0;
        ucur[nx+1][0] = 0.0;
        uold[0][ny+1] = 0.0;
        ucur[0][ny+1] = 0.0;
        uold[ny+1][0] = 0.0;
        ucur[ny+1][0] = 0.0;
        uold[nx+1][ny+1] = 0.0;
        ucur[ny+1][ny+1] = 0.0;

        // Impose Dirichlet BCs on the top
        for (j=1; j<=ny; j++){
            uold[0][j] = 0.0;
            ucur[0][j] = 0.0;
        }

        // Impose Dirichlet BCs on the bottom
        for (j=1; j<=ny; j++){
            uold[nx+1][j] = 0.0;
            ucur[nx+1][j] = 0.0;
        }

        // Impose Dirichlet BCs on the left
        for (i=1; i<=ny; i++){
            uold[i][0] = 0.0;
            ucur[i][0] = 0.0;
        }

        // Impose Dirichlet BCs on the right
        for (i=1; i<=ny; i++){
            uold[i][ny+1] = 0.0;
            ucur[i][ny+1] = 0.0;
        }

        // compute the next time level
        for (i=1; i<=nx; i++){
            for (j=1; j<=ny; j++){
                unew[i][j] = 2*ucur[i][j] + uold[i][j] 
                             + csq*dtdxsq*(ucur[i+1][j] - 2.0*ucur[i][j] + ucur[i-1][j]
                                         + ucur[i][j+1] - 2.0*ucur[i][j] + ucur[i][j-1]);
            }
        }

        // update the arrays
        for (i=1; i<=nx; i++){
            for (j=1; j<=ny; j++){
                uold[i][j] = ucur[i][j];
                ucur[i][j] = unew[i][j];
            }
        }
    printf("Finsihed computing timestep: %i\n", k); 
    }
    
    // free memory. There's a corresponding free to every malloc
    // XXX: Does this work?
    free(*uold);
    free(*ucur);
    free(*unew);
    free(uold);
    free(ucur);
    free(unew);
    printf("All done!\n");
}
