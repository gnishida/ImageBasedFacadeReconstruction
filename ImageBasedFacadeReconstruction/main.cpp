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
	for (auto it = boost::filesystem::directory_iterator(dir); it != boost::filesystem::directory_iterator(); ++it) {
		if (boost::filesystem::is_directory(it->path())) continue;

		// read an image
		cv::Mat img = cv::imread(dir.string() + it->path().filename().string());

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
