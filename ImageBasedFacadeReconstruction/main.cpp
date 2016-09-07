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
	bool align_windows = true;

	boost::filesystem::path dir("../testdata/");
	boost::filesystem::path dir_subdiv("../subdivision/");
	boost::filesystem::path dir_win("../windows/");
	boost::filesystem::path dir_results("../results/");
	boost::filesystem::path dir_grad("../grad/");
	boost::filesystem::path dir_fft("../fft/");
	for (auto it = boost::filesystem::directory_iterator(dir); it != boost::filesystem::directory_iterator(); ++it) {
		if (boost::filesystem::is_directory(it->path())) continue;

		// read an image
		cv::Mat img = cv::imread(dir.string() + it->path().filename().string());

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
			fs::getSplitLines2(Ver, 4, y_split);
			fs::getSplitLines2(Hor, 4, x_split);

			fs::outputImageWithHorizontalAndVerticalGraph(img, Ver, y_split, Hor, x_split, dir_grad.string() + it->path().filename().string(), 1);
		}

		/*
		cv::Mat grayImg;
		cvutils::grayScale(img, grayImg);
		cv::Mat sobelx;
		cv::Sobel(grayImg, sobelx, CV_32F, 1, 0);
		sobelx = cv::abs(sobelx);
		cv::Mat sobely;
		cv::Sobel(grayImg, sobely, CV_32F, 0, 1);
		sobely = cv::abs(sobely);

		cv::Mat gradx;
		cv::Mat grady;
		cv::reduce(sobely, gradx, 0, CV_REDUCE_SUM);
		cv::reduce(sobelx, grady, 1, CV_REDUCE_SUM);
		cv::blur(gradx, gradx, cv::Size(11, 11));
		cv::blur(grady, grady, cv::Size(11, 11));
		*/
		/*
		std::ofstream out(dir_fft.string() + it->path().filename().string() + "_hor.txt");
		for (int c = 0; c < gradx.cols; ++c) {
			out << gradx.at<float>(0, c) << std::endl;
		}
		out.close();
		std::ofstream out2(dir_fft.string() + it->path().filename().string() + "_ver.txt");
		for (int r = 0; r < grady.rows; ++r) {
			out2 << grady.at<float>(r, 0) << std::endl;
		}
		out2.close();
		*/

		//cvutils::outputImageWithHorizontalAndVerticalGraph(img, grady, gradx, dir_grad.string() + it->path().filename().string(), cvutils::LOCAL_MINIMUM, 4, 1);
		
		/*
		std::ofstream out(dir_fft.string() + it->path().filename().string() + "_hor.txt");
		for (int c = 0; c < gradx.cols; ++c) {
			out << gradx.at<float>(0, c) << std::endl;
		}
		out.close();
		*/

		// subdivide the facade into tiles and windows
		std::vector<float> x_split;
		std::vector<float> y_split;
		std::vector<std::vector<fs::WindowPos>> win_rects;
		fs::subdivideFacade(img, align_windows, y_split, x_split, win_rects);

		std::cout << it->path().filename().string() << std::endl;

		// visualize the segmentation and save it to files
		fs::outputFacadeStructure(img, y_split, x_split, dir_subdiv.string() + it->path().filename().string(), 1);
		fs::outputFacadeAndWindows(img, y_split, x_split, win_rects, dir_win.string() + it->path().filename().string());
		fs::outputWindows(y_split, x_split, win_rects, dir_results.string() + it->path().filename().string());
	}

	return 0;
}
