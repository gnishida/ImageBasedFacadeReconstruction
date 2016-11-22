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

		// read an image
		cv::Mat img = cv::imread(dir.string() + it->path().filename().string());

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
			cv::Canny(gray_img, edge_img, 50, 120);
			cv::imwrite(dir_edge.string() + it->path().filename().string(), edge_img);
		}

		// draw grad curve
		{
			cv::Mat_<float> Ver;
			cv::Mat_<float> Hor;
			fs::computeVerAndHor2(img, Ver, Hor);

			// smoothing
			cv::blur(Ver, Ver, cv::Size(11, 11));
			cv::blur(Hor, Hor, cv::Size(11, 11));

			// Facadeのsplit linesを求める
			std::vector<float> x_split;
			std::vector<float> y_split;
			fs::getSplitLines(Ver, 0.2, y_split);
			fs::getSplitLines(Hor, 0.2, x_split);

			fs::outputImageWithHorizontalAndVerticalGraph(img, Ver, y_split, Hor, x_split, dir_grad.string() + it->path().filename().string(), 1);
		}



		// subdivide the facade into tiles and windows
		std::vector<float> x_split;
		std::vector<float> y_split;
		std::vector<std::vector<fs::WindowPos>> win_rects;
		fs::subdivideFacade(img, align_windows, y_split, x_split, win_rects);

		std::cout << it->path().filename().string() << std::endl;

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
	}

	return 0;
}
