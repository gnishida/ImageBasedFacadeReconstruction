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
#include "EdgeDetection.h"

int main() {
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
		cv::Mat gray_img;
		cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

#if 0
		{
			cv::Mat_<float> Ver;
			cv::Mat_<float> Hor;
			fs::computeVerAndHor2(img, Ver, Hor);

			std::ofstream out(dir_curve.string() + it->path().filename().string() + ".txt", std::ofstream::out);
			for (int i = 0; i < Ver.rows; ++i) {
				out << Ver(i, 0) << std::endl;
			}
			out.close();
		}


		// edge images
		{
			cv::Mat gray_img;
			cvutils::grayScale(img, gray_img);
			cv::Mat edge_img;
			cv::Canny(gray_img, edge_img, 80, 240);
			cv::imwrite(dir_edge.string() + it->path().filename().string(), edge_img);
		}
#endif




		// subdivide the facade into tiles and windows
		std::vector<float> x_splits;
		std::vector<float> y_splits;
		std::vector<std::vector<fs::WindowPos>> win_rects;
		fs::subdivideFacade(it->path().filename().string(), gray_img, num_floors[it->path().filename().string()], align_windows, y_splits, x_splits, win_rects);

#if 0
		fs::outputFacadeStructure(img, y_splits, x_splits, dir_subdiv.string() + it->path().filename().string(), cv::Scalar(0, 255, 255), 3);

		// draw grad curve
		{
			// average floor height
			float average_floor_height = (float)img.rows / num_floors[it->path().filename().string()];

			// compute kernel size
			int kernel_size = average_floor_height / 6;
			if (kernel_size % 2 == 0) kernel_size++;

			// blur the image according to the average floor height
			if (kernel_size > 1) {
				cv::GaussianBlur(gray_img, gray_img, cv::Size(kernel_size, kernel_size), kernel_size);
			}

			cv::Mat_<float> SV_max;
			cv::Mat_<int> h_max;
			fs::computeSV(gray_img, SV_max, h_max, cv::Range(average_floor_height * 0.8, average_floor_height * 1.5));
			cv::Mat_<float> SH_max;
			cv::Mat_<int> w_max;
			fs::computeSH(gray_img, SH_max, w_max, cv::Range(average_floor_height * 0.4, average_floor_height * 2.4));


			cv::Mat_<float> Ver;
			cv::Mat_<float> Hor;
			fs::computeVerAndHor2(gray_img, Ver, Hor);

			// smooth Ver and Hor
			if (kernel_size > 1) {
				cv::blur(Ver, Ver, cv::Size(kernel_size, kernel_size));
				cv::blur(Hor, Hor, cv::Size(kernel_size, kernel_size));
			}

			fs::outputFacadeStructure(img, SV_max, Ver, h_max, y_splits, SH_max, Hor, w_max, x_splits, dir_grad.string() + it->path().filename().string(), cv::Scalar(0, 255, 255), 3);
		}
#endif


#if 0
		// visualize the segmentation and save it to files
		fs::outputFacadeStructure(img, y_split, x_split, dir_subdiv.string() + it->path().filename().string(), cv::Scalar(0, 255, 255), 3);
		fs::outputFacadeAndWindows(img, y_split, x_split, win_rects, dir_win.string() + it->path().filename().string(), cv::Scalar(0, 255, 255), 3);

		if (resize) {
			for (int i = 0; i < win_rects.size(); ++i) {
				for (int j = 0; j < win_rects[i].size(); ++j) {
					win_rects[i][j].left = win_rects[i][j].left * output_size.width / img.cols;
					win_rects[i][j].right = win_rects[i][j].right * output_size.width / img.cols;
					win_rects[i][j].top = win_rects[i][j].top * output_size.height / img.rows;
					win_rects[i][j].bottom = win_rects[i][j].bottom * output_size.height / img.rows;
				}
			}
			for (int i = 0; i < x_split.size(); ++i) {
				x_split[i] = x_split[i] * output_size.width / img.cols;
			}
			for (int i = 0; i < y_split.size(); ++i) {
				y_split[i] = y_split[i] * output_size.height / img.rows;
			}
		}
		fs::outputWindows(y_split, x_split, win_rects, dir_results.string() + it->path().filename().string(), cv::Scalar(0, 0, 0), 1);
#endif
	}

	return 0;
}
