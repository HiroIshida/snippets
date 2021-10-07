//extern void* newHoge(int idx, double* arr);
//extern int operation(void* obj, int a);
#include "wrapper.h"
#include "stdio.h"
int main(){
  int i=0;
  double a[10];
  void* obj = newHoge(i, a);
  int b = operation(obj, 3);
}
