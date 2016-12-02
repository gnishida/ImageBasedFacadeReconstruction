/**
* 論文Image-based procedural modeling of facadesに基づいて、
* Facadeを分割する。
* 5/24に第一回の報告をしたが、いろいろ難しいことが分かった。
* まず、MIではうまくsymmetryをキャプチャできないケースがある。
*
* @author Gen Nishida
* @version 1.0
*/

#include <algorithm>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <opencv2/opencv.hpp>
#include "CVUtils.h"
#include "Utils.h"
#include <time.h>
#include "FacadeSegmentation.h"
#include <list>
#include <boost/filesystem.hpp>

int main() {
# if 0
	boost::filesystem::path dir("../rolling/");
	for (auto it = boost::filesystem::directory_iterator(dir); it != boost::filesystem::directory_iterator(); ++it) {
		if (boost::filesystem::is_directory(it->path())) continue;

		cv::Mat img = cv::imread((std::string("../rolling/") + it->path().filename().string()).c_str());
		cv::Mat gray_img;
		cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

		cv::Mat_<float> Ver, Hor;
		fs::computeVerAndHor2(gray_img, Ver, Hor, 0);

		fs::outputImageWithHorizontalAndVerticalGraph(img, Ver, Hor, std::string("../rolling_grad/") + it->path().filename().string());

		cv::Mat edge_img;
		cv::Canny(img, edge_img, 30, 100);
		cv::imwrite((std::string("../rolling_edge/") + it->path().filename().string()).c_str(), edge_img);

		cv::Mat line_img = img.clone();
		std::vector<cv::Vec4i> lines;
		cv::HoughLinesP(edge_img, lines, 1, CV_PI / 180, 20, 3, 1);
		for (size_t i = 0; i < lines.size(); ++i) {
			float theta = atan2(lines[i][3] - lines[i][1], lines[i][2] - lines[i][0]) / CV_PI * 180;
			if (abs(theta) < 10 || abs(theta) > 170) {
				cv::line(line_img, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 255, 255), 3, 8);
			}
		}
		cv::imwrite((std::string("../rolling_line/") + it->path().filename().string()).c_str(), line_img);
	}
#endif


#if 1
	bool align_windows = false;
	bool resize = false;
	cv::Size output_size(227, 227);

	// read the #floors file
	std::ifstream in("floors.txt");
	std::map<std::string, int> num_floors;
	while (!in.eof()) {
		std::string filename;
		int num;
		in >> filename >> num;
		num_floors[filename] = num;

		if (filename == "") break;
	}

	boost::filesystem::path dir("../testdata/");
	//boost::filesystem::path dir("../testdata2/");
	boost::filesystem::path dir_curve("../curve/");
	boost::filesystem::path dir_subdiv("../subdivision/");
	boost::filesystem::path dir_win("../windows/");
	boost::filesystem::path dir_results("../results/");
	boost::filesystem::path dir_grad("../grad/");
	boost::filesystem::path dir_edge("../edge/");
	for (auto it = boost::filesystem::directory_iterator(dir); it != boost::filesystem::directory_iterator(); ++it) {
		if (boost::filesystem::is_directory(it->path())) continue;

		std::cout << it->path().filename().string() << std::endl;

		// read an image
		cv::Mat img = cv::imread(dir.string() + it->path().filename().string());

		// gray scale
		//cv::Mat gray_img;
		//cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

		// subdivide the facade into tiles and windows
		std::vector<float> x_splits;
		std::vector<float> y_splits;
		std::vector<std::vector<fs::WindowPos>> win_rects;
		fs::subdivideFacade(img, num_floors[it->path().filename().string()], align_windows, y_splits, x_splits, win_rects);

		// grad image
		{
			cv::Mat gray_img;
			cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

			// average floor height
			float average_floor_height = (float)img.rows / num_floors[it->path().filename().string()];

			// compute kernel size
			int kernel_size = average_floor_height / 8;
			if (kernel_size % 2 == 0) kernel_size++;

			// blur the image according to the average floor height
			cv::Mat blurred_gray_img;
			if (kernel_size > 1) {
				cv::GaussianBlur(gray_img, blurred_gray_img, cv::Size(kernel_size, kernel_size), kernel_size);
			}
			else {
				blurred_gray_img = gray_img.clone();
			}

			cv::Range h_range = cv::Range(average_floor_height * 0.8, average_floor_height * 1.5);
			cv::Range w_range = cv::Range(average_floor_height * 0.7, average_floor_height * 2.4);

			// compute Ver and Hor
			cv::Mat_<float> Ver, Hor;
			fs::computeVerAndHor2(blurred_gray_img, Ver, Hor, 0.0);

			// smooth Ver and Hor
			if (kernel_size > 1) {
				cv::blur(Ver, Ver, cv::Size(kernel_size, kernel_size));
				cv::blur(Hor, Hor, cv::Size(kernel_size, kernel_size));
			}

			/*
			cv::Mat_<float> SV_max;
			cv::Mat_<int> h_max;
			fs::computeSV(blurred_gray_img, SV_max, h_max, h_range);
			cv::Mat_<float> SH_max;
			cv::Mat_<int> w_max;
			fs::computeSH(blurred_gray_img, SH_max, w_max, w_range);
			*/

			//fs::outputFacadeStructure(img, SV_max, Ver, h_max, y_splits, SH_max, Hor, w_max, x_splits, dir_grad.string() + it->path().filename().string(), cv::Scalar(0, 255, 255), 1);
			fs::outputImageWithHorizontalAndVerticalGraph(img, Ver, y_splits, Hor, x_splits, std::string("../grad/") + it->path().filename().string(), 1);
		}

		// subdivision image
		fs::outputFacadeStructure(img, y_splits, x_splits, dir_subdiv.string() + it->path().filename().string(), cv::Scalar(0, 255, 255), 3);
		
		// window image
		fs::outputFacadeAndWindows(img, y_splits, x_splits, win_rects, dir_win.string() + it->path().filename().string(), cv::Scalar(0, 255, 255), 3);

	}

#endif

	return 0;
}
