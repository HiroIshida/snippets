#include <stdlib.h>
#include <iostream>
using namespace std;

void make_matrix(double*** mat_, int n1, int n2){
    double** mat = (double**)malloc(n1 * sizeof(double*));
    for(int i=0; i<n1; i++){
        mat[i] = (double*)malloc(n2 * sizeof(double));
    }
    cout << "matrix" << endl;
    for(int i=0; i<n1; i++){
        for(int j=0; j<n2; j++){
            mat[i][j] = (double)(n1*i + j);
            cout << mat[i][j] << endl;
        }
    }
    *mat_ = mat;
}

int main(){
    double** mat;
    make_matrix(&mat, 3, 3);
    cout << "checking" << endl;
    cout << mat[2][2] << endl;
    free(mat);
}

