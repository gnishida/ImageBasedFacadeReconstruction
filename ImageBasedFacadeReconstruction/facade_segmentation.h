#include <vector>
#include <opencv2/opencv.hpp>
#include "ibfr.h"

using namespace std;

float MI(const cv::Mat& R1, const cv::Mat& R2);
void computeSV(const cv::Mat& img, cv::Mat_<float>& SV_max, cv::Mat_<float>& h_max, const pair<int, int>& h_range);
void computeSH(const cv::Mat& img, cv::Mat_<float>& SH_max, cv::Mat_<float>& w_max, const pair<int, int>& w_range);
void computeVerAndHor(const cv::Mat& img, cv::Mat_<float>& Ver, cv::Mat_<float>& Hor, float sigma);
bool subdivideTile(const cv::Mat& tile, const cv::Mat& edges, int min_size, int tile_margin, WindowPos& winpos);
void findBestHorizontalSplitLines(const cv::Mat& img, const cv::Mat_<float>& Ver, float min_interval, float max_interval, vector<int>& y_split);
void findBestVerticalSplitLines(const cv::Mat& img, const cv::Mat_<float>& Hor, float min_interval, float max_interval, vector<int>& x_split);
void getSplitLines(const cv::Mat_<float>& mat, vector<float>& split_positions);
void refineSplitLines(vector<float>& split_positions);
void refine(vector<float>& y_split, vector<float>& x_split, vector<vector<WindowPos>>& winpos);
void align(const vector<float>& y_split, const vector<float>& x_split, vector<vector<WindowPos>>& winpos);