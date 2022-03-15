#include<opencv2/opencv.hpp>  
#include"/usr/local/include/opencv2/core/core.hpp"
#include "/home/lxy/opencv-3.4.8/opencv_contrib-3.4.8/modules/face/include/opencv2/face/facerec.hpp"
#include "/usr/local/include/opencv2/core.hpp"
#include "/usr/local/include/opencv2/face.hpp"
#include "/usr/local/include/opencv2/highgui/highgui.hpp"
 #include "/usr/local/include/opencv2/imgproc/imgproc.hpp"
#include <math.h>  
#include <stdlib.h>
#include <string>
#include <vector>
#include "time.h"
#include "CvxText.h"
//使用void read_csv()这个函数必须的三个头文件
#include <iostream>  
#include <fstream>  
#include <sstream> 
 
using namespace std;
using namespace cv;
using namespace cv::face;
static int ToWchar(char* &src, wchar_t* &dest, const char *locale = "zh_CN.utf8")
{
    if (src == NULL) {
        dest = NULL;
        return 0;
    }

    // 根据环境变量设置locale
    setlocale(LC_CTYPE, locale);

    // 得到转化为需要的宽字符大小
    int w_size = mbstowcs(NULL, src, 0) + 1;

    // w_size = 0 说明mbstowcs返回值为-1。即在运行过程中遇到了非法字符(很有可能使locale
    // 没有设置正确)
    if (w_size == 0) {
        dest = NULL;
        return -1;
    }

    //wcout << "w_size" << w_size << endl;
    dest = new wchar_t[w_size];
    if (!dest) {
        return -1;
    }

    int ret = mbstowcs(dest, src, strlen(src)+1);
    if (ret <= 0) {
        return -1;
    }
    return 0;
}
RNG g_rng(12345);
Ptr<FaceRecognizer> model;
 
int Predict(Mat src_image)  //识别图片
{
	Mat face_test;
	int predict = 0;
	//截取的ROI人脸尺寸调整
	if (src_image.rows >= 120)
	{
		//改变图像大小，使用双线性差值
		resize(src_image, face_test, Size(92, 112));
 
	}
	//判断是否正确检测ROI
	if (!face_test.empty())
	{
		//测试图像应该是灰度图  
		predict = model->predict(face_test);
	}
	cout << predict << endl;
	return predict;
}







int main()
{
	VideoCapture cap(0);    //打开默认摄像头  
	if (!cap.isOpened())
	{
		return -1;
	}
	Mat frame;
	Mat gray;
	//这个分类器是人脸检测所用
	CascadeClassifier cascade;
	bool stop = false;
	//训练好的文件名称，放置在可执行文件同目录下  
	cascade.load("/home/lxy/opencv-3.4.8/data/haarcascades/haarcascade_frontalface_alt2.xml");//感觉用lbpcascade_frontalface效果没有它好，注意哈！要是正脸
 
	model = FisherFaceRecognizer::create();
	//1.加载训练好的分类器
	model->read("/home/lxy/test/test1/MyFaceFisherModel.xml");// opencv2用load
		
		
		
		
		








//3.利用摄像头采集人脸并识别
	while (1)
	{
		cap >> frame;
  
		vector<Rect> faces(0);//建立用于存放人脸的向量容器
		
		cvtColor(frame, gray, CV_RGB2GRAY);//测试图像必须为灰度图
		
		equalizeHist(gray, gray); //变换后的图像进行直方图均值化处理  
		//检测人脸
		cascade.detectMultiScale(gray, faces,
			1.1, 4, 0
			//|CV_HAAR_FIND_BIGGEST_OBJECT  
			| CV_HAAR_DO_ROUGH_SEARCH,
			//| CV_HAAR_SCALE_IMAGE,
			Size(30, 30), Size(500, 500));
		Mat* pImage_roi = new Mat[faces.size()];    //定以数组
		Mat face;
		Point text_lb;//文本写在的位置
		//框出人脸
		 char*  str;
		for (int i = 0; i < faces.size(); i++)
		{
			pImage_roi[i] = gray(faces[i]); //将所有的脸部保存起来
			text_lb = Point(faces[i].x, faces[i].y);
			if (pImage_roi[i].empty())
				continue;
			switch (Predict(pImage_roi[i])) //对每张脸都识别
			{

			case 24:str = (char *)"2021091203006李晓阳";break;
			default: str = (char *)"error"; break;
			}
			Scalar color = Scalar(g_rng.uniform(100, 255), g_rng.uniform(10, 255), g_rng.uniform(20, 255));//所取的颜色任意值
			rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), color, 1, 8);//放入缓存

			    CvxText text("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"); //指定字体
    cv::Scalar size1{ 30, 0.5, 0.1, 0 }; // (字体大小, 无效的, 字符间距, 无效的 }

    text.setFont(nullptr, &size1, nullptr, 0);
    wchar_t *w_str;
    ToWchar(str,w_str);
    text.putText(frame, w_str, cv::Point(50,100), cv::Scalar(0, 0, 255));
  cv::resize(frame, frame, cv::Size(800,800));
   // cv::imshow("demo", frame);
   // cv::waitKey(0);
			//putText(frame, str, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));//添加文字
		}
 
		delete[]pImage_roi;
		imshow("face", frame);
		waitKey(200);	
	}
 
	return 0;
}
