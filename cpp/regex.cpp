#include <regex>
#include <string>
#include <iostream>
using namespace std; 
int main(){
  std::string subject = "package://fetch_description/meshes/base_link.obj";
  std::regex integer_expr("(\\+|-)?[[:digit:]]+");
  // [^/] means that all chracter except /.
  std::regex re("package://([^/]*)/(.*)");
  std::smatch match;
  std::string result;
  if (std::regex_search(subject, match, re) && match.size() > 1) {
    result = match.str(1);
  }
  cout << result << endl;
}

