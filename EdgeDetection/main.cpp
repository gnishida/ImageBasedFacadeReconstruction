#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main() {
	cv::Mat img = cv::imread("../testdata/11.png");

	cv::Mat gray_img;
	cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

	cv::Mat edges;
	cv::Canny(gray_img, edges, 100, 300);

	cv::imwrite("11.png", edges);

	return 0;
}