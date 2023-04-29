#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags
#include <string>
#include <stdexcept>
#include <vector>
#include <array>
#include <iostream>


struct Trimesh
{
  std::vector<std::array<double, 3>> vertices;
  std::vector<std::array<size_t, 3>> faces;

  friend std::ostream& operator<<(std::ostream& os, const Trimesh& trimesh) {
    for(const auto & v : trimesh.vertices) {
     os << "v " << v[0] << ", " << v[1] << ", " << v[2] << std::endl;
    }
    for(const auto & f : trimesh.faces) {
     os << "f " << f[0] << ", " << f[1] << ", " << f[2] << std::endl;
    }
    return os;
  }
};

Trimesh merge_trimeshes(const std::vector<Trimesh>& trimeshes) {
  Trimesh trimesh_new;
  size_t idx_head = 0;
  for(const auto & trimesh : trimeshes) {
    for(const auto & v : trimesh.vertices) {
      trimesh_new.vertices.push_back(v);
    }
    for(auto f : trimesh.faces) {
      f[0] += idx_head;
      f[1] += idx_head;
      f[2] += idx_head;
      trimesh_new.faces.push_back(f);
    }
    idx_head += trimesh.vertices.size();
  }
  return trimesh_new;
}


int main() { 
  Assimp::Importer importer;
  const std::string file_path = "/home/h-ishida/.cache/robot_descriptions/jaxon_description/meshes/BODY.dae";
  const aiScene* scene = importer.ReadFile(file_path, 0);

  if(scene->mNumMeshes == 0){
    std::cout << scene->mNumMeshes << std::endl;
    throw std::runtime_error("hoge");
  }

  std::vector<Trimesh> trimeshes;

  for(size_t i = 0; i < scene->mNumMeshes; ++i) {
    const auto mesh = scene->mMeshes[i];

    auto trimesh = Trimesh();

    for(size_t j = 0; j < mesh->mNumVertices; ++j) {
      double x = mesh->mVertices[j].x;
      double y = mesh->mVertices[j].y;
      double z = mesh->mVertices[j].z;
      trimesh.vertices.push_back(std::array<double, 3>{x, y, z});
    }

    for(size_t j = 0; j < mesh->mNumFaces; ++j) {
      size_t idx1 = mesh->mFaces[j].mIndices[0] + 1;
      size_t idx2 = mesh->mFaces[j].mIndices[1] + 1;
      size_t idx3 = mesh->mFaces[j].mIndices[2] + 1;
      trimesh.faces.push_back(std::array<size_t, 3>{idx1, idx2, idx3});
    }
    trimeshes.push_back(trimesh);
  }

  const auto trimesh_merged = merge_trimeshes(trimeshes);
  std::cout << trimesh_merged << std::endl;
}

