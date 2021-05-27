double test(double a){
  return a * a;
}

double test_func_pointer(void (*hoge)(double a)){
    hoge(114514);
}
