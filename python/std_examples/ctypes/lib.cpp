#include<cmath>
#include<algorithm>
#include<limits>

extern "C" {
    void* make_boxes(double* xmin, double* xmax, double* ymin, double* ymax, int n);
    void delete_boxes(void* boxes);
    double signed_distance(double x, double y, void* boxes);
    void signed_distance_batch(double* x, double* y, double* dist, int n, void* boxes);
}

struct Box {
    double xmin, xmax, ymin, ymax;

    double signed_distance(double x, double y) {
        double dx = std::max({0.0, xmin - x, x - xmax});
        double dy = std::max({0.0, ymin - y, y - ymax});
        if (x >= xmin && x <= xmax && y >= ymin && y <= ymax) {
            return -std::min({x - xmin, xmax - x, y - ymin, ymax - y});
        }
        return std::sqrt(dx*dx + dy*dy);
    }
};

struct Boxes {
  Box* boxes;
  int n;
  ~Boxes() {delete[] boxes;}
};

void* make_boxes(double* xmin, double* xmax, double* ymin, double* ymax, int n) {
  auto bs = new Boxes;
  bs->boxes = new Box[n];
  bs->n = n;
  for (int i = 0; i < n; ++i) {
    bs->boxes[i] = {xmin[i], xmax[i], ymin[i], ymax[i]};
  }
  return bs;
}

double signed_distance(double x, double y, void* boxes) {
  Boxes* bs = static_cast<Boxes*>(boxes);
  double min_dist = std::numeric_limits<double>::infinity();
  for (int i = 0; i < bs->n; ++i) {
    min_dist = std::min(min_dist, bs->boxes[i].signed_distance(x, y));
  }
  return min_dist;
}

void signed_distance_batch(double* x, double* y, double* dist, int n, void* boxes) {
  Boxes* bs = static_cast<Boxes*>(boxes);
  for (int i = 0; i < n; ++i) {
    dist[i] = signed_distance(x[i], y[i], bs);
  }
}

void delete_boxes(void* boxes) {
  delete static_cast<Boxes*>(boxes);
}
