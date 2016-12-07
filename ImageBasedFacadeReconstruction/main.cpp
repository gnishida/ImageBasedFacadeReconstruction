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

	// read the #floors file
	std::ifstream in("floors_columns.txt");
	std::map<std::string, std::pair<int, int>> params;
	while (!in.eof()) {
		std::string filename;
		int v1, v2;
		in >> filename >> v1 >> v2;

		params[filename] = std::make_pair(v1, v2);

		if (filename == "") break;
	}

	std::ofstream tile_out("tiles.txt");


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

		// floor height / column width
		float average_floor_height = (float)img.rows / params[it->path().filename().string()].first;
		float average_column_width = (float)img.cols / params[it->path().filename().string()].second;

		// gray scale
		//cv::Mat gray_img;
		//cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

		// subdivide the facade into tiles and windows
		std::vector<float> x_splits;
		std::vector<float> y_splits;
		std::vector<std::vector<fs::WindowPos>> win_rects;
		fs::subdivideFacade(img, average_floor_height, average_column_width, align_windows, y_splits, x_splits, win_rects);

		// grad image
		{
			cv::Mat gray_img;
			cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

			// compute kernel size
			int kernel_size_V = average_floor_height / 8;
			if (kernel_size_V % 2 == 0) kernel_size_V++;
			int kernel_size_H = average_column_width / 8;
			if (kernel_size_H % 2 == 0) kernel_size_H++;

			// blur the image according to the average floor height
			cv::Mat blurred_gray_img;
			if (kernel_size_V > 1) {
				cv::GaussianBlur(gray_img, blurred_gray_img, cv::Size(kernel_size_V, kernel_size_V), kernel_size_V);
			}
			else {
				blurred_gray_img = gray_img.clone();
			}

			cv::Range h_range = cv::Range(average_floor_height * 0.8, average_floor_height * 1.5);
			cv::Range w_range = cv::Range(average_column_width * 0.6, average_column_width * 1.3);

			// compute Ver and Hor
			cv::Mat_<float> Ver, Hor;
			fs::computeVerAndHor2(blurred_gray_img, Ver, Hor, 0.0);

			// smooth Ver and Hor
			if (kernel_size_V > 1) {
				cv::blur(Ver, Ver, cv::Size(kernel_size_V, kernel_size_V));
			}
			if (kernel_size_H > 1) {
				cv::blur(Hor, Hor, cv::Size(kernel_size_H, kernel_size_H));
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

		// tile images
		char base_name[256];
		sscanf(it->path().filename().string().c_str(), "%s.png", base_name);
		int tile_cnt = 0;
		for (int i = 0; i < y_splits.size() - 1; ++i) {
			for (int j = 0; j < x_splits.size() - 1; ++j) {
				int x1 = x_splits[j];
				int x2 = x_splits[j + 1];
				int y1 = y_splits[i];
				int y2 = y_splits[i + 1];

				if ((y_splits.size() - 1) * (x_splits.size() - 1) < 20 || (tile_cnt < 20 && rand() % 4 == 0)) {

					char file_name[256];
					sprintf(file_name, "%s_%d_%d.png", base_name, i, j);
					cv::Mat tile_img(img, cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1));
					cv::Mat resized_tile_img;
					cv::resize(tile_img, resized_tile_img, cv::Size(227, 227));
					cv::imwrite((std::string("../tiles/") + file_name).c_str(), resized_tile_img);

					tile_out << file_name << "\t\n";

					tile_cnt++;
				}
			}
		}
	}
	tile_out.close();

	return 0;
}
