#include <regex>
#include <string>
#include <iostream>
using namespace std; 

// because regex is so slow to compile, it's better to write small 
// utils like below if can be simple
std::string extract_path_rospackage1(const std::string& meshpath_){
  std::string meshpath = meshpath_;
  std::string trash;
  for(auto it = meshpath.begin(); it < meshpath.end(); it++){
    trash.push_back(*it);
    if(trash.compare("package://")==0){
      meshpath.erase(meshpath.begin(), it+1);
      cout << meshpath << endl;
      return meshpath;
    }
  }
  // raise exception
}

std::string extract_path_rospackage2(const std::string& meshpath_){
  // or more simply just remove fast 10 chracters
  std::string meshpath = meshpath_;
  meshpath.erase(0, 10);
  cout << meshpath << endl;
  return meshpath;
}

void get_path(const string& str)
{
  size_t found;
  found=str.find_last_of("/\\");
  cout << " folder: " << str.substr(0,found) << endl;
  cout << " file: " << str.substr(found+1) << endl;
}


int main(){
  std::string subject = "package://fetch_description/meshes/base_link.obj";
  extract_path_rospackage2(subject);

  std::string subject1 = "../robot/fetch_description/fetch.urdf";
  get_path(subject1);
}

int doit_by_regex(){
  std::string subject = "package://fetch_description/meshes/base_link.obj";
  // [^/] means that all chracter except /.
  std::regex re("package://(.*)/([^/]*)");
  std::smatch match;
  std::string result;
  if (std::regex_search(subject, match, re) && match.size() > 1) {
    result = match.str(1); // str(0) returns the whole string, so basically we never call str(0)
  }
  cout << result << endl;

  std::string subject1 = "../robot/fetch_description/fetch.urdf";
  std::smatch match1;
  std::regex re1("(.+)/([^/]+\\.urdf)"); // to escape a meta caracter, use two backslash instead of single backslash
  if (std::regex_search(subject1, match1, re1) && match1.size() > 0) {
    result = match1.str(1); // str(0) returns the whole string, so basically we never call str(0)
    cout << result << endl;
  }
}
