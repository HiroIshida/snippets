#include <regex>
#include <string>
#include <iostream>
using namespace std; 
int main(){
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

