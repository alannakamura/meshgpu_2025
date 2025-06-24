__device__ float g1_mw(int m, int n, float *position, int i)
{
    float g1 = 0, temp;
    int j;

    for(j=m-1;j<n;j++)
    {
        temp = powf(position[i*n+j], (n-m));
//         printf("%e ", temp);
//         printf("%d ", j);
        temp -= 0.5;
//         printf("%0.8lf ", temp);
        temp -= ((float)j/(2*n));
//         printf("%lf ", temp);
        temp *= temp;
        temp *= -10;
        temp = 1-expf(temp);
//         printf("%lf ", temp);

        g1 += temp;
    }
//     printf("\n");
    return g1+1;
}

__device__ void mw1(float *position, int *position_dim, float *fitness, int i)
{
    float g=0, l, c, pi = 3.141592;

    g = g1_mw(2, position_dim[0], position, i);

    fitness[i*2+0] =  position[i*position_dim[0]+0];
    fitness[i*2+1] =  g*(1-0.85*fitness[i*2+0]/g);

    l = sqrt(2.0)*fitness[i*2+1] - sqrt(2.0) * fitness[i*2+0];
    c = 1- fitness[i*2+1] - fitness[i*2+0]+0.5*pow(sin(2*pi*l), 8);

    if(c<0)
    {
        printf("%d c=%lf l=%lf g=%lf\n", i, c, l, g);
//         fitness[i*2+0] = 1e5;
//         fitness[i*2+1] = 1e5;
    }
}

__global__ void function(int *func_n, float *position, int *position_dim, float *fitness)
{
    int i = threadIdx.x;

    if(func_n[0] == 31)
    {
        mw1(position, position_dim, fitness, i);
    }
}