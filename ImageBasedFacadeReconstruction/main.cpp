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
		fs::subdivideFacade(it->path().filename().string(), img, num_floors[it->path().filename().string()], align_windows, y_splits, x_splits, win_rects);



		// window images
		fs::outputFacadeAndWindows(img, y_splits, x_splits, win_rects, dir_win.string() + it->path().filename().string(), cv::Scalar(0, 255, 255), 3);

	}

	return 0;
}
