#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main(){
  // &= operator: get min values
  {
    Mat img1(3, 3, CV_8UC1, Scalar(0));
    Mat img2(3, 3, CV_8UC1, Scalar(255));
    img1 &= img2;
    cout << img1 << endl; // all 0
  }
  {
    Mat img1(3, 3, CV_8UC1, Scalar(255));
    Mat img2(3, 3, CV_8UC1, Scalar(0));
    img1 &= img2;
    cout << img1 << endl; // all 0
  }

  {
    Mat img1(3, 3, CV_8UC1, Scalar(255));
    Mat img2(3, 3, CV_8UC1, Scalar(255));
    img1 &= img2;
    cout << img1 << endl; // all 255
  }

  {
    Mat img1(3, 3, CV_8UC1, Scalar(150));
    Mat img2(3, 3, CV_8UC1, Scalar(255));
    img1 &= img2;
    cout << img1 << endl; // all 150
  }

  // |= operator: get max values
  {
    Mat img1(3, 3, CV_8UC1, Scalar(150));
    Mat img2(3, 3, CV_8UC1, Scalar(255));
    img1 |= img2;
    cout << img1 << endl; // all 255
  }


}
