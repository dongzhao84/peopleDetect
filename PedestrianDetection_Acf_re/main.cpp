//
//  main.cpp
//  PedestrianDetection_Acf
//
//  Created by 董昭 on 17/8/28.
//  Copyright © 2017年 董昭. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include<vector>
#include <sstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml.hpp>

#include "PedDetector.h"
using namespace std;
using namespace cv;

#define DataSet "/Users/Dzhaoooo/Desktop/ImageData"  //测试数据集路径（正例）
// ChnFtrsAdaboost.cpp : 定义控制台应用程序的入口点。FPDW版

//获取指定文件夹下的所有文件名列表
vector<string> getAllFiles(){
    DIR *dp;
    struct dirent *dirp;
    vector<string> filenames;
    
    dp=opendir(DataSet);
    while((dirp=readdir(dp))!=NULL){
        if(strcmp(dirp->d_name, ".")!=0 && strcmp(dirp->d_name, "..")!=0 &&strcmp(dirp->d_name, ".DS_Store")!=0){
            filenames.push_back(dirp->d_name);
        }
    }
    return filenames;
    
}

//图像放大
Mat resizeImage(string imageName){
    
    double scale=2;
    Mat src=imread(imageName);
    Size ResImgSiz = Size(src.cols*scale, src.rows*scale);
    //采用立方插值法进行图像缩放
    Mat result1=Mat(ResImgSiz,src.type());
    resize(src, result1, ResImgSiz, CV_INTER_CUBIC);
    
    Mat BlurImage;
    medianBlur(result1, BlurImage, 3);
    
    //    Mat ShapeImage;
    //    Mat kernel=(Mat_<float>(3,3)<<0,-1,0,-1,5,-1,0,-1,0);
    //    filter2D(BlurImage, ShapeImage, BlurImage.depth(), kernel);
    
    
    //    return result1;
    return BlurImage;
}
//
////行人检测代码
//void acfDetect(string filename){
//    PedDetector pd;
//    pd.loadStrongClassifier("/Users/Dzhaoooo/Desktop/PedestrianDetection_Acf/PedestrianDetection_Acf/ClassifierOut.txt");
//    cout<<"正在处理图片："<<filename<<endl;
//    Mat src=resizeImage(ImgName);
//    IplImage img = IplImage(src);
//    //    const char *srcImg=ImgName.data();
//    //    IplImage *img = cvLoadImage(srcImg);
//    CvMat *ReMat=NULL;
//    pd.Detection_FPDW(&img, &ReMat, 3);     //行人检测接口函数
//    pd.show_detections(&img, ReMat);     //显示检测结果
//    
//    //    pd.Detection_FPDW(img, &ReMat, 3);     //行人检测接口函数
//    //    pd.show_detections(img, ReMat);     //显示检测结果
//    
//    string saveName;
//    stringstream temp;
//    string indexStr;
//    temp<<index;
//    temp>>indexStr;
//    saveName="/Users/Dzhaoooo/Desktop/Result/"+indexStr+".jpg";
//    const char * result=saveName.data();
//    cvSaveImage(result,&img);
//    
//}



int main(int argc, const char * argv[])
{
    string imgPath="/Users/Dzhaoooo/Desktop/ImageData/fuck.jpg";
    PedDetector pd;
    pd.loadStrongClassifier("/Users/Dzhaoooo/Desktop/PedestrianDetection_Acf/PedestrianDetection_Acf/ClassifierOut.txt");
    cout<<"正在处理图片："<<imgPath<<endl;
    Mat src=imread(imgPath);
    cout<<src.cols*2;
    Size ResImgSiz = Size(src.cols*2, src.rows*2);
    //采用立方插值法进行图像缩放
    Mat result1=Mat(ResImgSiz,src.type());
    resize(src, result1, ResImgSiz, CV_INTER_CUBIC);
    
    Mat BlurImage;
    medianBlur(result1, BlurImage, 3);
    
    IplImage img = IplImage(BlurImage);
    CvMat *ReMat=NULL;
    pd.Detection_FPDW(&img, &ReMat, 3);     //行人检测接口函数
    string resultInfo=pd.show_detections(&img, ReMat);     //显示检测结果
    int index=0;
    string saveName;
    stringstream temp;
    string indexStr;
    temp<<index;
    temp>>indexStr;
    saveName="/Users/Dzhaoooo/Desktop/Result/"+indexStr+".jpg";
    const char * result=saveName.data();
    cvSaveImage(result,&img);


//    vector<string> filenames=getAllFiles();
//    size_t size=filenames.size();
//    for(int i=0;i<size;i++)
//    {
//        ofstream outfile;
//        String filname = (String)filenames[i];
//        //
//        outfile.open("/Users/Dzhaoooo/Desktop/Json/"+filname+".json");
//        acfDetect(filname, i);
//        ifstream infile("/Users/Dzhaoooo/Desktop/RectInform.json");
//        string line;
//        outfile<<"\"filename\":"<<filname<<"\",\"objects\":[";
//        int temp = 0;
//        while(getline(infile, line)){
//            if(temp>0)
//                outfile<<",";
//            outfile<<line;
//            temp++;
//            
//        }
//        outfile<<"]}"<<endl;
//        outfile.close();
//        infile.close();
//    }
    
    
    cout<<"the info of oic is .........."<<resultInfo<<endl;
    return 0;
    
}

