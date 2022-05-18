#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <limits>
#include <vector>
#include <iostream>
#include <math.h>

inline double cross_product2d(Eigen::Vector2d && vec1, Eigen::Vector2d vec2) {
  return vec1.x() * vec2.y() - vec2.x() * vec1.y();
}

class Box
{
  struct Edge2d
  {
    size_t idx_start;
    size_t idx_end;
  };

public:
  Box(double x, double y, double yaw, double length, double width)
  {
    // create points
    std::vector<Eigen::Vector2d> points(4);
    points[0] << +0.5 * length, +0.5 * width;
    points[1] << -0.5 * length, +0.5 * width;
    points[2] << -0.5 * length, -0.5 * width;
    points[3] << 0.5 * length, -0.5 * width;

    Eigen::Vector2d trans;
    trans << x, y;

    const auto rot = Eigen::Rotation2Dd(yaw);
    for (Eigen::Vector2d & p : points) {
      p = rot * p + trans;
    }

    double angle_min = + std::numeric_limits<double>::infinity();
    double angle_max = - std::numeric_limits<double>::infinity();
    for (Eigen::Vector2d & p : points) {
      const double angle = std::atan2(p.y(), p.x());
      angle_min = std::min(angle_min, angle);
      angle_max = std::max(angle_max, angle);
    }

    // check if edge is visible
    std::vector<Edge2d> visible_edges;
    for (size_t i = 0; i < 4; ++i) {
      const Eigen::Vector2d p0 = points[i];
      const size_t i_next = (i == 3 ? 0 : i + 1);
      const Eigen::Vector2d p1 = points[i_next];
      const bool is_visible = cross_product2d(p1 - p0, p0) > 0.0;
      if (is_visible) {
        visible_edges.push_back(Edge2d{i, i_next});
      }
    }

    points_ = points;
    visible_edges_ = visible_edges;
    angle_min_ = angle_min;
    angle_max_ = angle_max;
  }

  double collision_distance(double angle) const {

    const auto inverse_mat = [](const Eigen::Vector2d & v, const Eigen::Vector2d & w) {
      Eigen::Matrix2d mat;
      mat << w.y(), -w.x(), -v.y(), v.x();
      const auto det = mat.determinant();
      mat /= det;
      return mat;
    };

    Eigen::Vector2d v, w;
    v << cos(angle), sin(angle);

    double dist = std::numeric_limits<double>::infinity();
    for (const auto & edge : visible_edges_) {
      // s * v + t * w = p1
      const auto & p0 = points_[edge.idx_start];
      const auto & p1 = points_[edge.idx_end];
      const auto && w = p1 - p0;
      const auto st = inverse_mat(v, w) * p1;

      const double s = st.x();
      const double t = st.y();
      const bool is_hit = (0.0 < t && t < 1.0);
      if (is_hit) {
        dist = s;
      }
    }
    return dist;
  }

  std::vector<Eigen::Vector2d> points_;
  std::vector<Edge2d> visible_edges_;
  double angle_min_;
  double angle_max_;
};

int main(){
  {
    const auto b = Box(2.0, 2.0, 0.0, 2.0, 2.0);

    assert(abs(b.angle_min_ - std::atan2(1., 3.)) < 1e-5);
    assert(abs(b.angle_max_ - std::atan2(3., 1.)) < 1e-5);

    {
      auto val = b.collision_distance(M_PI * 0.25);
      assert(abs(val - sqrt(2.0)) < 1e-5);
    }

    {
      auto val = b.collision_distance(atan2(1, 2.0));
      assert(abs(val - sqrt(5.0)) < 1e-5);
    }

    {
      auto val = b.collision_distance(atan2(1, 3.0) -1e-8);
      assert(val = std::numeric_limits<double>::infinity());
    }

    {
      auto val = b.collision_distance(atan2(1, 3.0) +1e-8);
      assert(abs(val - sqrt(10.0)) < 1e-2);
    }
  }

}
