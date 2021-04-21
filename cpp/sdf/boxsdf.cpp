#include<iostream>
#include<Eigen/Core>
using namespace Eigen;

/* fetched from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}
*/

double boxsdf(const Eigen::Vector2d& pos, const Eigen::Vector2d& b)
{
  Eigen::Vector2d q = pos.cwiseAbs() - b;
  Eigen::Vector2d left_max(std::max(q(0), 0.0), std::max(q(1), 0.0));
  double left = left_max.norm();
  double right = std::min(q.maxCoeff(), 0.0);
  return left + right;
}
int main(){
  // testing
  {
    Vector2d p(1.5, 0);
    Vector2d b(1, 2);
    std::cout << (boxsdf(p, b) == 0.5) << std::endl; 
  }
  {
    Vector2d p(0, 0);
    Vector2d b(1, 2);
    std::cout << (boxsdf(p, b) == -1.0) << std::endl; 
  }
  {
    Vector2d p(0, 1.5);
    Vector2d b(1, 2);
    std::cout << (boxsdf(p, b) == -0.5) << std::endl; 
  }
  {
    Vector2d p(0, 2.5);
    Vector2d b(1, 2);
    std::cout << (boxsdf(p, b) == +0.5) << std::endl; 
  }

  {
    int N = 1000000;
    Vector2d p(0, 2.5);
    Vector2d b(1, 2);
    clock_t start = clock();
    for(int i=0; i<N; i++){
      boxsdf(p, b);
    }
    clock_t end = clock();
    std::cout << (end - start) << " [micro sec]" << std::endl; 
  }

  //benchmarking
}

