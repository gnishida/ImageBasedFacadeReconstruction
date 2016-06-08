#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "ibfr.h"

using namespace std;

void outputFacadeStructureV(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& h_max, const vector<int>& y_set, const string& filename, int lineWidth);
void outputFacadeStructureV(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& h_max, const string& filename, int lineWidth);
void outputFacadeStructureH(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& w_max, const vector<int>& x_set, const string& filename, int lineWidth);
void outputFacadeStructureH(const cv::Mat& img, const cv::Mat_<float>& S_max, const cv::Mat_<float>& w_max, const string& filename, int lineWidth);
void outputFacadeStructure(const cv::Mat& img, const cv::Mat_<float>& SV_max, const cv::Mat_<float>& h_max, const cv::Mat_<float>& SH_max, const cv::Mat_<float>& w_max, const vector<int>& y_set, const vector<int>& x_set, const string& filename, int lineWidth);
void outputFacadeStructure(const cv::Mat& img, const cv::Mat_<float>& SV_max, const cv::Mat_<float>& h_max, const cv::Mat_<float>& SH_max, const cv::Mat_<float>& w_max, const string& filename, int lineWidth);
void outputFacadeStructure(const cv::Mat& img, const vector<int>& y_set, const vector<int>& x_set, const string& filename, int lineWidth);
void outputFacadeAndWindows(const cv::Mat& img, const vector<int>& y_split, const vector<int>& x_split, const vector<vector<cv::Rect>>& window_rects, const string& filename);

void outputFacadeSegmentation(const cv::Mat& img, const vector<int>& y_split, const vector<Tile>& floors, const string& filename);
