#include <fcl/broadphase/broadphase_collision_manager.h>
#include <fcl/common/types.h>
#include <fcl/math/triangle.h>
#include <fcl/narrowphase/collision.h>
#include <fcl/narrowphase/distance.h>
#include <memory>

typedef fcl::BVHModel<fcl::OBBRSSf> Model;

template <typename S>
void loadOBJFile(const char* filename, std::vector<fcl::Vector3<S>>& points, std::vector<fcl::Triangle>& triangles)
{

  FILE* file = fopen(filename, "rb");
  if(!file)
  {
    std::cerr << "file not exist" << std::endl;
    return;
  }

  bool has_normal = false;
  bool has_texture = false;
  char line_buffer[2000];
  while(fgets(line_buffer, 2000, file))
  {
    char* first_token = strtok(line_buffer, "\r\n\t ");
    if(!first_token || first_token[0] == '#' || first_token[0] == 0)
      continue;

    switch(first_token[0])
    {
    case 'v':
      {
        if(first_token[1] == 'n')
        {
          strtok(nullptr, "\t ");
          strtok(nullptr, "\t ");
          strtok(nullptr, "\t ");
          has_normal = true;
        }
        else if(first_token[1] == 't')
        {
          strtok(nullptr, "\t ");
          strtok(nullptr, "\t ");
          has_texture = true;
        }
        else
        {
          S x = (S)atof(strtok(nullptr, "\t "));
          S y = (S)atof(strtok(nullptr, "\t "));
          S z = (S)atof(strtok(nullptr, "\t "));
          points.emplace_back(x, y, z);
        }
      }
      break;
    case 'f':
      {
        fcl::Triangle tri;
        char* data[30];
        int n = 0;
        while((data[n] = strtok(nullptr, "\t \r\n")) != nullptr)
        {
          if(strlen(data[n]))
            n++;
        }

        for(int t = 0; t < (n - 2); ++t)
        {
          if((!has_texture) && (!has_normal))
          {
            tri[0] = atoi(data[0]) - 1;
            tri[1] = atoi(data[1]) - 1;
            tri[2] = atoi(data[2]) - 1;
          }
          else
          {
            const char *v1;
            for(int i = 0; i < 3; i++)
            {
              // vertex ID
              if(i == 0)
                v1 = data[0];
              else
                v1 = data[t + i];

              tri[i] = atoi(v1) - 1;
            }
          }
          triangles.push_back(tri);
        }
      }
    }
  }
}
      

int main() {
  std::vector<fcl::Vector3f> V;
  std::vector<fcl::Triangle> T;
  loadOBJFile("../obj.obj", V, T);

  std::shared_ptr<Model> geom1 = std::make_shared<Model>();
  geom1->beginModel();
  geom1->addSubModel(V, T);
  geom1->endModel();
  auto pose1 = fcl::Transform3f::Identity();
  fcl::CollisionObjectf* obj1 = new fcl::CollisionObjectf(geom1, pose1);

  std::shared_ptr<Model> geom2 = std::make_shared<Model>();
  geom2->beginModel();
  geom2->addSubModel(V, T);
  geom2->endModel();
  auto pose2 = fcl::Transform3f::Identity();
  pose2.translation() = fcl::Vector3f(3., 3, 3);
  fcl::CollisionObjectf* obj2 = new fcl::CollisionObjectf(geom2, pose2);

  fcl::DistanceRequestf request;
  fcl::DistanceResultf result;

  fcl::distance(obj1, obj2, request, result);
  std::cout << result.min_distance << std::endl;
}
