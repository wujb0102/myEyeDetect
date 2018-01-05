#include <opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include <math.h>
#include <sstream>
#include <fstream>
#include<opencv2/core.hpp>
#include<opencv2/face.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/face/facerec.hpp>
#include <iostream>

using namespace cv;
using namespace cv::face;
using namespace std;

//创建和返回一个归一化的图像矩阵
static Mat norm_0_255(InputArray _src) 		//InputArray这个接口类可以是Mat、Mat_<T>、Mat_<T, m, n> 、vector<T> 等等
{
	Mat src = _src.getMat();
	Mat dst;
	switch(src.channels()){
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst); 	//将源对象完全拷贝给目的对象
		break;
	}
	return dst;
}

//使用CSV文件去读图像和标签，主要使用stringstream和getline方法
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& lables, char separator = ';')
{
	std::ifstream file(filename.c_str(), ifstream::in); 	//从硬盘到内存读入CSV文件
	if(!file){
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while(getline(file, line)){ 	//从输入流中读入一行到string line
		stringstream liness(line); 	//将string转换为流
		getline(liness, path, separator); 	//从流中读入一行到path中
		getline(liness, classlabel); 	//从流中读入一行到classbel中
		if(!path.empty() && !classlabel.empty()){
			images.push_back(imread(path, 0)); 	//在容器尾部插入对象
			lables.push_back(atoi(classlabel.c_str())); 	//先将字符串转换为数字再插入到容器尾部
		}
	}
}

int main()
{
	//读取你的CSV文件路径
	//string fn_csv = string(argv[1]);
	string fn_csv = "at.txt";
	
	//2个容器用来存放图像数据和对应的标签
	vector<Mat> images;
	vector<int> labels;
	//读取数据，如果文件不合法就会出错
	//输入文件名已经有了
	try{
		read_csv(fn_csv, images, labels);
	}
	catch(cv::Exception& e){
	//	ceer << "Error opening file \"" << fn_csv << "\".Reason: " << e.msg << endl;
		//文件有问题，不能继续执行，退出
		
		exit(1);
	}
	if(images.size() <= 1){
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}
	for(int i = 0; i < images.size(); i++){
		if(images[i].size() != Size(92, 112)){
			cout << i << endl;
			cout << images[i].size() << endl;
		}
	}

	//下面几行代码仅仅是从你的数据集中移除最后一张图片
	//[gm: 自然那这里需要根据自己的需要修改， 他这里简化了很多问题]
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
//	images.pop_back();
//	labels.pop_back();
	//下面几行创建了一个特征脸模型用于人脸识别，
	//通过CSV文件读取的图像和标签来训练它。
	//T这里是一个完整的PCV转换
	//如果你只想保留10个猪成分，使用如下代码
	//	cv::creatEigenFaceRecognizer(10);
	//如果你还希望使用置信度阈值来初始化，使用以下语句：
	//	cv::creatEigenFaceRecognizer(10, 123.0);
	//
	//如果你使用所有特征并且使用一个阈值，使用以下语句
	//	cv::creatEigenFaceRecognizer(0,123.0);
	

	Ptr<BasicFaceRecognizer> model = EigenFaceRecognizer::create();
	model->train(images, labels);
	model->save("MyFacePCAMedel.xml");

	Ptr<BasicFaceRecognizer> model1 = FisherFaceRecognizer::create();
	model1->train(images, labels);
	model1->save("MyFaceFisherModel.xml");

	Ptr<LBPHFaceRecognizer> model2 = LBPHFaceRecognizer::create();
	model2->train(images,labels);
	model2->save("MyFaceLBPHModel.xml");

	//下面对测试图像进行预测，predicteLabel是预测标签结果
	int predictedLabel = model->predict(testSample);
	int predictedLabel1 = model1->predict(testSample);
	int predictedLabel2 = model2->predict(testSample);

	//还有一种调试方式，可以获得结果同时得到阈值：
	//	int predictedLabel = -1;
	//	double confidence = 0.0;
	//	model->predict(testSanple, predictedLabel, confidence);
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	string result_message1 = format("Predicted class = %d / Actual class = %d.", predictedLabel1, testLabel);
	string result_message2 = format("Predicted class = %d / Actual class = %d.", predictedLabel2, testLabel);
	cout << result_message << endl;
	cout << result_message1 << endl;
       	cout << result_message2 << endl;
	getchar();
	waitKey(0);
	return 0;
}










































