#include<stdlib.h>

int main(){
    double*** a;
    int n = 3;
    int m = 3;
    int l = 3;

    a = (double***)malloc(n * sizeof(double**));
    double**** pts = &a;
    for(int i=0; i<n; i++){
        (*pts)[i] = (double**)malloc(m * sizeof(double*));
        for(int j=0; j<m; j++){
            (*pts)[i][j] = (double*)malloc(l * sizeof(double));
        }
    }
}
