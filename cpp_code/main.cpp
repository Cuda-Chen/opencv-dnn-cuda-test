#include <iostream>
#include <string>
#include <ctime>
#include <chrono>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"
/*
using namespace std;
using namespace cv;
using namespace dnn;
*/

int main()
{
	double confidenceThreshold = 0.5;
	double NMSThreshold = 0.4;
	int numClasses = 80;

	std::string modelPath = "../yolos/yolov3.weights";
	std::string configPath = "../yolos/yolov3.cfg";
	std::string inputImg = "../yolos/dog.jpg";
	auto net = cv::dnn::readNet(modelPath, configPath);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

	cv::Mat image = cv::imread(inputImg);
	cv::Mat blob = cv::dnn::blobFromImage(image, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
	auto outputNames = net.getUnconnectedOutLayersNames();
	std::vector<cv::Mat> detections;

	// warmup
	for(int i = 0; i < 3; i++)
	{
		net.setInput(blob);
		net.forward(detections, outputNames);
	}

	// benckmark
	auto start = std::chrono::steady_clock::now();
	for(int i = 0; i < 100; i++)
	{
		net.setInput(blob);
		net.forward(detections, outputNames);
	}
	auto end = std::chrono::steady_clock::now();

	std::chrono::milliseconds timeTaken = std::chrono::duration_cast<std::chrono::milliseconds>((end - start) / 100.0);
	std::cout << "Time per inference: " << timeTaken.count() << " ms" << std::endl
		<< "FPS: " << 1000.0 / (timeTaken.count()) << std::endl;

	return 0;
}
