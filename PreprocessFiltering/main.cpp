#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/opencv.hpp>

int main() {
	cv::Mat img = cv::imread("../testdata/03.png");
	cv::imshow("original", img);

	cv::Mat filtered;
	cv::bilateralFilter(img, filtered, -1, 40, 20);

	cv::imshow("filtered", filtered);

	
	cv::Mat edges;
	cv::Canny(filtered, edges, 100, 250);
	cv::imshow("edges", edges);

	cv::waitKey();



	return 0;
}