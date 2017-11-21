//
//  PedDetector.cpp
//  PedestrianDetection_Acf
//
//  Created by 董昭 on 17/8/28.
//  Copyright © 2017年 董昭. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <fstream>
#include <iostream>
#include "PedDetector.h"
#include "DollarMex.h"
#include <algorithm>
#include <math.h>
using namespace std;


PedDetector::PedDetector(void)
{
    ChnFtrsChannelNum = 10;   //积分通道数，默认为10,3个LUV颜色通道+1个梯度幅值+6个梯度方向
    nPerOct = 8;     //需要建立的金字塔层数
    m_Shrink = 4;      //特征矩形框的边长
    OverLapRate = 0.65;   //覆盖率阈值，剔除掉覆盖率大于此值的检测框
    num = 0;      //弱分类器个数
    softThreshold = NULL;
    FeaID = NULL;     //决策树分类器每个节点所对应的特征编号，两层决策树，一共三个节点
    DCtree = NULL;       //双层决策树分类器
}


PedDetector::~PedDetector(void)
{
    Release();
}

void PedDetector::Release()
{
    if (DCtree != NULL)
        delete []DCtree;
    
    if (FeaID != NULL){
        delete[] FeaID;
        FeaID = NULL;
    }
}


//将基于MatLab代码训练好的分类器导入
bool PedDetector::loadStrongClassifier(const char *pFilePath)
{
    FILE *fs=fopen(pFilePath, "r");
    Release();
    /*	this->num = WeakNum;*/
    fscanf(fs, "%d", &this->num);  //读取fs指向的文件的第一个整型数，并将其赋值给num，即弱分类器个数为2048
    
    //objectSize为模型大小
    this->objectSize.width = 64, this->objectSize.height = 128;
    this->softThreshold = -1;
    this->FeaID = new int[this->num*3];   //分类器特征编号对应的数组，个数为分类器个数*3
    //划窗的步长为4*4
    this->xStride = 4;
    this->yStride = 4;
    this->scaleStride = 1.08; // 尺度在的步长没有采用此固定值
    this->nms = NMS_METHOD_OutWin;  //nms采用基于贪心策略
    // 读取2048个弱分类器决策树上各个节点所对应的特征编号
    for (int i=0; i<this->num; i++)
    {
        for (int j=0; j<3; j++)
        {
            fscanf(fs, "%d ", &this->FeaID[i*3+j]);
        }
        int temp1, temp2, temp3, temp4;
        fscanf(fs, "%d %d %d %d ", &temp1, &temp2, &temp3, &temp4);
    }
    // 读取2048个弱分类器决策树上不同节点所对应的决策阈值
    this->DCtree=new WeakClassifier[this->num];
    for (int i=0; i<this->num; i++)
    {
        for (int j=0; j<3; j++)
        {
            fscanf(fs, "%f ", &this->DCtree[i].threshold[j]);
        }
        float temp1, temp2, temp3, temp4;
        fscanf(fs, "%f %f %f %f ", &temp1, &temp2, &temp3, &temp4);
    }
    // 读取2048个弱分类器决策树上不同节点所对应的权值
    for (int i=0; i<this->num; i++)
    {
        for (int j=0; j<7; j++)
        {
            fscanf(fs, "%f ", &this->DCtree[i].hs[j]);
        }
    }
    fclose(fs);
    // 初始化特征位置索引
    FeaIn = new FtrIndex[5120];  //5120=(64/4)*(128/4)*10
    int m=0;
    CvRect rect;
    rect.width = (this->objectSize.width)/m_Shrink;
    rect.height = (this->objectSize.height)/m_Shrink;
    for( int z=0; z<ChnFtrsChannelNum; z++ )
        for( int c=0; c<rect.width; c++ )
            for( int r=0; r<rect.height; r++ )
            {
                FeaIn[m].Channel=z;
                FeaIn[m].x=c;
                FeaIn[m++].y=r;
            }
    
    return true;
}

float PedDetector::StrongClassify(CvMat *pChnFtrsData)
{
    float* tempChnFtrs;
    tempChnFtrs = pChnFtrsData->data.fl;
    float ans=0.0f;
    for (int i=0; i<this->num; i++)   //i从0到2047
    {
        //取出第i个弱分类器决策树对应的叶子节点的最终权重并相加，最终返回所有弱分类器的叶子节点的权重之和
        ans+=DCtree[i].Classify(tempChnFtrs+3*i);
        if (ans<-1) return ans;
    }
    return ans;
}

string Trans(int ss)
{
    string str;
    stringstream st;
    st << ss;
    st >> str;
    return str;
}

/*
 将行人检测结果打印到原始图像中并显示
 输入：
 pImg 原始图像
 pAllResults 检测结果
 color 检测框显示颜色 */
string PedDetector::show_detections(IplImage *pImg, CvMat *pAllResults, CvScalar color)
{
//    ofstream ofile;
    
    
    if (pAllResults == NULL)
    {
        return 0;
    }
    CvScalar FondColor;
    CvFont font;        //用于屏幕输出
    char str[100];
    cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 2);   //字体结构的初始化
    FondColor = CV_RGB(255,255,255);   //FondColor为白色
    int i;
    string info;
//    ofile.open("/Users/Dzhaoooo/Desktop/RectInform.json");
    for (i=0; i<pAllResults->rows; i++)
    {
        CvScalar tmp = cvGet1D(pAllResults, i);    //返回数组pAllResults的下标i处对应的值
        tmp.val[0] = tmp.val[0];
        tmp.val[1] = tmp.val[1];
        tmp.val[2] = tmp.val[2];
        CvRect rectDraw;
        rectDraw.width = 40;
        rectDraw.height = 100;
        if (tmp.val[3] > 0)
            color = CV_RGB(0, 255, 0);
        else
            color = CV_RGB(200, 100, 100);
        {
            // 显示分数
//            sprintf(str, "%.4f", tmp.val[3]);
            //str:打印到图片上的内容 cvPoint:字符串在图片上打印的原点 &font:字体属性变量 FondColor：字体颜色
            //            cvPutText(pImg, str,
            //                      cvPoint(int(tmp.val[0]-rectDraw.width/2*tmp.val[2]), int(tmp.val[1]-rectDraw.height/2*tmp.val[2])+12),
            //                      &font, FondColor);
            // 显示检测结果框
            cvRectangle(pImg,
                        cvPoint(int(tmp.val[0]-rectDraw.width/2*tmp.val[2]), int(tmp.val[1]-rectDraw.height/2*tmp.val[2])),
                        cvPoint(int(tmp.val[0]+rectDraw.width/2*tmp.val[2]), int(tmp.val[1]+rectDraw.height/2*tmp.val[2])),
                        color, 2);
            int minx=int(tmp.val[0]-rectDraw.width/2*tmp.val[2]);
            string xmin=Trans(minx);
            int maxy=int(tmp.val[1]-rectDraw.height/2*tmp.val[2]);
            string ymax=Trans(maxy);
            int maxx=int(tmp.val[0]+rectDraw.width/2*tmp.val[2]);
            string xmax=Trans(maxx);
            int miny=int(tmp.val[1]+rectDraw.height/2*tmp.val[2]);
            string ymin=Trans(miny);
            
            info+=xmin+","+ymin+","+xmax+","+ymax+";";
            //实际顺序是 xmin ymax xmax ymin
           
//            
//            ofile<<"{\"xmin\":\""<<minx<<"\","<<"\"ymin\":\""<<miny<<"\","<<"\"xmax\":\""<<maxx<<"\","<<"\"ymax\":\""<<maxy<<"\"}"<<endl;
            
            
        }
    }
    return info;
//    ofile.close();
//    return true;
}

/*
 行人检测接口函数，基于FPDW方法，详见BMVC2010 - the fastest pedestrian detector in the west）
 输入：
 pImgInput 待检测图像
 pAllResults 存储检测结果，存储格式为（每行数据按 检测框中心x坐标、检测框中心y坐标、检测框相对于行人模型的缩放倍数、最终检测得分 排列）
 UpScale 检测行人的最大尺度上限，以128*64大小为基准 */

bool PedDetector::Detection_FPDW(IplImage *pImgInput, CvMat **pAllResults, float UpScale)
{
    CvMat ***ChnFtrs; // 存储积分通道特征
    float ***ChnFtrs_float; // 平滑前的特征
    float scaleStridePerOctave = 2.0f; // FPDW中每个Octive的尺度宽度
    float nowScale; // 当前尺度
    CvRect rect, PicRect; // rect是检测框大小，PicRect是当前尺度下的整幅图像大小
    float ans; // 结果分数
    float *FeaData = new float[this->num*3]; // 特征
    int shrink_rate = m_Shrink; // 稀疏率
    CvScalar *tmpScalar; // 检测结果格式
    
    int step, t;
    int t_AllIntegral = 0;
    float *Scales; // 图像金字塔不同层所对应的缩放倍数
    int itemp1, itemp2;
    
    CvMemStorage *memory; // 检测结果队列内存
    CvSeq *seqDetections; // 检测结果队列
    memory = cvCreateMemStorage();
    seqDetections = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvScalar), memory);
    rect.width = 64, rect.height = 128;
    
    // 图像的存储格式转换，并转换到LUV颜色空间
    int h = (pImgInput->height/shrink_rate)*shrink_rate, w = (pImgInput->width/shrink_rate)*shrink_rate;
    IplImage *pImg = cvCreateImage(cvSize(w, h), pImgInput->depth, pImgInput->nChannels);
    cvSetImageROI(pImgInput, cvRect(0, 0, w, h));
    cvCopy(pImgInput,pImg);
    //    cvCopyImage(pImgInput, pImg);
    cvResetImageROI(pImgInput);
    //int h = pImg->height, w = pImg->width;
    int d=pImg->nChannels;
    int ChnBox = h*w;
    unsigned char *data = new unsigned char[h*w*3];
    IplImage *imgB = cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);
    IplImage *imgG = cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);
    IplImage *imgR = cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);
    cvSplit(pImg, imgB, imgG, imgR, NULL);
    for (int i=0; i<ChnBox; i++){
        data[i] = imgR->imageData[i];
        data[ChnBox+i] = imgG->imageData[i];
        data[2*ChnBox+i] = imgB->imageData[i];
    }
    cvReleaseImage(&imgR);
    cvReleaseImage(&imgG);
    
    float *luvImg = rgbConvert(data, ChnBox, 3, 2, 1.0/255);
    delete[] data;
    
    //计算图像金字塔层数
    nowScale=1.0;
    int nScales = min((int)(nPerOct*(log(min((double)pImg->width/64, (double)pImg->height/128))/log(2.0))+1),
                      (int)(nPerOct*log(UpScale)/log(2.0)+1));
    ChnFtrs = new CvMat**[nScales];
    Scales = new float[nScales];
    // 计算每层金字塔所对应的缩放倍数
    for (int i=0; i<nScales; i++){
        Scales[i] = pow(2.0, (double)(i+1)/nPerOct);
    }
    int *Octives = new int[nScales];
    int nOctives = 0;
    bool *isChnFtrs = new bool[nScales]; // 标记每层金字塔是否计算
    memset(isChnFtrs, 0, nScales*sizeof(bool));
    // 标记Octives，先计算Octives，然后其他尺度由相邻的Octives估计出来（FPDW方法，详见BMVC2010 - the fastest pedestrian detector in the west）
    while (nOctives*nPerOct<nScales){
        Octives[nOctives] = nOctives*nPerOct;
        nOctives++;
    }
    
    int *NearOctive = new int[nScales]; // 标记每层图像金字塔的近似源
    int iTemp = 0;
    for (int i=1; i<nOctives; i++){
        for (int j=iTemp; j<=(Octives[i-1]+Octives[i])/2; j++){
            NearOctive[j] = Octives[i-1];
        }
        iTemp = (Octives[i-1]+Octives[i])/2+1;
    }
    for (int i=iTemp; i<nScales; i++) NearOctive[i] = Octives[nOctives-1];
    //先计算Octives的积分通道特征
    for (int i=0; i<nOctives; i++){
        isChnFtrs[Octives[i]] = true;
        ChnFtrs[Octives[i]] = new CvMat *[ChnFtrsChannelNum];
        GetChnFtrs(luvImg, h, w, shrink_rate, Scales[Octives[i]], ChnFtrs[Octives[i]]);
    }
    // 估计出图像金字塔参数 lambdas
    float lambdas[10] = {0};
    if (nOctives<2){
        for (int i=3; i<10; i++) lambdas[i] = 0.1158;
    }
    else{
        for (int i=3; i<10; i++){
            float f0, f1;
            CvScalar tempS;
            tempS = cvSum(ChnFtrs[Octives[0]][i]);
            f0 = tempS.val[0]/(ChnFtrs[Octives[0]][i]->width*ChnFtrs[Octives[0]][i]->height);
            tempS = cvSum(ChnFtrs[Octives[1]][i]);
            f1 = tempS.val[0]/(ChnFtrs[Octives[1]][i]->width*ChnFtrs[Octives[1]][i]->height);
            lambdas[i] = log(f1/f0)/log(2.0);
        }
    }
    
    // 根据Octives近似resample出其他层的图像金子塔
    for (int i=0; i<nScales; i++){
        if (isChnFtrs[i]) continue;
        int hNow = (int)(h/Scales[i]/shrink_rate+0.5);
        int wNow = (int)(w/Scales[i]/shrink_rate+0.5);
        int h0 = ChnFtrs[NearOctive[i]][0]->height;
        int w0 = ChnFtrs[NearOctive[i]][0]->width;
        ChnFtrs[i] = new CvMat*[ChnFtrsChannelNum];
        for (int j=0; j<ChnFtrsChannelNum; j++){
            float ratio = pow(Scales[NearOctive[i]]/Scales[i], -lambdas[j]);
            ChnFtrs[i][j] = cvCreateMat(hNow, wNow, CV_32FC1);
            float_resample(ChnFtrs[NearOctive[i]][j]->data.fl, ChnFtrs[i][j]->data.fl, w0, wNow, h0, hNow, 1, ratio);
        }
    }
    for (int i=0; i<nScales; i++){
        for (int j=0; j<ChnFtrsChannelNum; j++){
            convTri1(ChnFtrs[i][j]->data.fl, ChnFtrs[i][j]->data.fl, ChnFtrs[i][j]->width, ChnFtrs[i][j]->height, 1, (float)2.0, 1);
        }
    }
    
    // AdaBoost分类计算过程
    //nScales为金字塔的层数
    for (step=0; step<nScales; step++)
    {
        //尺度金字塔结束条件
        if ((int)(pImg->width/Scales[step]+0.5f)<rect.width || (int)(pImg->height/Scales[step]+0.5f)<rect.height)
        {
            break;
        }
        CvSize ScaleSize = cvSize(ChnFtrs[step][0]->width*shrink_rate, ChnFtrs[step][0]->height*shrink_rate);
        
        //当前尺度下整幅图的宽和高
        PicRect.width = ChnFtrs[step][0]->width;
        PicRect.height = ChnFtrs[step][0]->height;
        // 密集滑窗操作
        for (PicRect.y = 0; PicRect.y+rect.height <= ScaleSize.height; PicRect.y += yStride)
        {
            for (PicRect.x = 0; PicRect.x+rect.width <= ScaleSize.width; PicRect.x += xStride)
            {
                rect.x=(PicRect.x/m_Shrink);
                rect.y=(PicRect.y/m_Shrink);
                
                //对于每个窗口的级联分类过程
                float score = 0.0;
                for (t=0; t<this->num; t++)
                {
                    for (int j=0; j<3; j++)
                    {
                        int temp;
                        temp=this->FeaID[t*3+j];
                        FeaData[t*3+j]=ChnFtrs[step][FeaIn[temp].Channel]->data.fl[(FeaIn[temp].y+rect.y)*PicRect.width+FeaIn[temp].x+rect.x];
                    }
                    score += this->DCtree[t].Classify(FeaData+t*3);
                    if (score<softThreshold) break;
                }
                if (score > 0.0)
                {
                    tmpScalar = (CvScalar *)cvSeqPush(seqDetections, NULL);
                    tmpScalar->val[0] = (PicRect.x + rect.width/2) * Scales[step];//检测框中心x坐标;
                    tmpScalar->val[1] = (PicRect.y + rect.height/2) * Scales[step];//检测框中心y坐标;
                    tmpScalar->val[2] = Scales[step];//检测框相对于行人模型的缩放倍数;
                    tmpScalar->val[3] = score ; // 检测分数
                }
            }
        }
    }
    cvReleaseImage(&pImg);
    // 释放积分通道特征内存
    for (step=0; step<nScales; step++){
        for (int i=0; i<ChnFtrsChannelNum; i++){
            cvReleaseMat(&ChnFtrs[step][i]);
        }
        delete[] ChnFtrs[step];
    }
    delete[] ChnFtrs;
    
    
    // non maximum suppression 非极大值抑制过程
    CvMat *pDetections = NULL;
    CvMat *pModes = NULL;
    //seqDetections为检测结果队列
    if (seqDetections->total > 0)
    {
        pDetections = cvCreateMat(seqDetections->total, 1, CV_64FC4);
        for (int i=0; i<seqDetections->total; i++)
        {
            tmpScalar = (CvScalar *)cvGetSeqElem(seqDetections, i);
            cvSet1D(pDetections, i, *tmpScalar);
        }
        
        if(nms==NMS_METHOD_OutWin){
            OutWin_NonMaxSuppression(pDetections, &pModes);
        }else{
            pModes = (CvMat *)cvClone(pDetections);
        }
//                if(nms==NMS_METHOD_MixScale){
//                    MixScale_NonMaxSuppression(pDetections,&pModes);
//                }else{
//                    pModes = (CvMat *)cvClone(pDetections);
//                }
        
        //        if (nms == NMS_METHOD_MaxGreedy)
        //            MaxGreedy_NonMaxSuppression(pDetections, OverLapRate, &pModes);
        //        else
        //        {
        //            //输出所有detection
        //            pModes = (CvMat *)cvClone(pDetections);
        //        }
    }
    cvReleaseMemStorage(&memory);
    cvReleaseMat(&pDetections);
    
    if (*pAllResults != NULL)
        cvReleaseMat(pAllResults);
    *pAllResults = pModes;
    delete[] luvImg;
    delete[] FeaData;
    delete[] isChnFtrs;
    delete[] Scales;
    delete[] Octives;
    delete[] NearOctive;
    return true;
}

// NMS非极大值抑制，运动贪心的办法，将检测结果按分数排序，从前至后按覆盖率筛选结果;
static int cmp_detection_by_score(const void *_a, const void *_b)
{
    double *a = (double *)_a;
    double *b = (double *)_b;
    if (a[3] > b[3]) // [0]:x, [1]:y, [2]:s, [3]:score
        return -1;
    if (a[3] < b[3])
        return 1;
    return 0;
}
// 获取两个检测框的相互覆盖率
double GetOverlapRate(double *D1, double *D2)
{
    double Pw, Ph;
    Pw = 41.0, Ph = 100.0;
    double xr1, yr1, xl1, yl1;
    double xr2, yr2, xl2, yl2;
    xl1 = D1[0]-D1[2]*Pw/2;
    xr1 = D1[0]+D1[2]*Pw/2;
    xl2 = D2[0]-D2[2]*Pw/2;
    xr2 = D2[0]+D2[2]*Pw/2;
    double ix = min(xr1, xr2) - max(xl1, xl2);
    if (ix<0) return -1;
    yr2 = D2[1]+D2[2]*Ph/2;
    yl1 = D1[1]-D1[2]*Ph/2;
    yr1 = D1[1]+D1[2]*Ph/2;
    yl2 = D2[1]-D2[2]*Ph/2;
    double iy = min(yr1, yr2) - max(yl1, yl2);
    if (iy<0) return -1;
    //覆盖率的计算为两检测框的面积交／两者面积的最小值
    return ix*iy/min((yr1-yl1)*(xr1-xl1), (yr2-yl2)*(xr2-xl2));
    
}

//融合尺度比信息的动态重合面积阈值
double DynamicHhreshold(double *D1,double *D2){
    double thr;
    double Pw, Ph;
    Pw = 41.0, Ph = 100.0;
    double yl1,yr1,yl2,yr2;
    double score1,score2;
    double height1,height2;
    score1=D1[3];
    score2=D2[3];
    yr2 = D2[1]+D2[2]*Ph/2;
    yl1 = D1[1]-D1[2]*Ph/2;
    yr1 = D1[1]+D1[2]*Ph/2;
    yl2 = D2[1]-D2[2]*Ph/2;
    height1=yr1-yl1;
    height2=yr2-yl2;
    if(score2/score1>=0.7){
        thr=0.65;
    }else{
        thr=min(0.65,0.65*pow((height2/(height1*0.75)), 2));
    }
    return thr;
}

//判断当前抑制窗口是否完全被被抑制窗口包围且两者检测分数差异极小

bool SaveOutWin(double *D1,double *D2){
    double Pw, Ph;
    Pw = 41.0, Ph = 100.0;
    double xr1,yr1,xl1,yl1;
    double xr2,yr2,xl2,yl2;
    double score1,score2,scoreRatio;
    xl1 = D1[0]-D1[2]*Pw/2;
    xr1 = D1[0]+D1[2]*Pw/2;
    xl2 = D2[0]-D2[2]*Pw/2;
    xr2 = D2[0]+D2[2]*Pw/2;
    yr2 = D2[1]+D2[2]*Ph/2;
    yl1 = D1[1]-D1[2]*Ph/2;
    yr1 = D1[1]+D1[2]*Ph/2;
    yl2 = D2[1]-D2[2]*Ph/2;
    score1=D1[3];
    score2=D2[3];
    scoreRatio=score2/score1;
    if((xl2<=xl1)&&(xr1<=xr2)&&(yl2<=yl1)&&(yr1<=yr2)&&(scoreRatio>0.85)){
        return true;
    }else{
        return false;
    }
    
}
void PedDetector::MaxGreedy_NonMaxSuppression(CvMat *pDetections, float overlap, CvMat **pModes)
{
    // 对于所有检测框按分数高低排序
    qsort((void *)pDetections->data.db, pDetections->rows, 4*sizeof(double), cmp_detection_by_score);
    int n = pDetections->rows;
    int ReTotal=0;
    bool *isHold = new bool[n];
    //将isHold数组的元素置换为1并返回isHold指针
    memset(isHold, 1, n*sizeof(bool));
    
    // 按分数从高到底，保留高分检测框，并剔除掉与保留窗口覆盖率超过overlap的低分值窗口（贪心策略）
    for (int i=0; i<n; i++){
        if (!isHold[i]) continue;
        ReTotal++;
        CvScalar Di;
        Di = cvGet1D(pDetections, i);
        for (int j=i+1; j<n; j++){
            double overlapRate;
            CvScalar Dj;
            Dj = cvGet1D(pDetections, j);
            overlapRate = GetOverlapRate(Di.val, Dj.val);
            if (overlapRate<0) continue;
            if (overlapRate>overlap) isHold[j] = false;
        }
    }
    *pModes = cvCreateMat(ReTotal, 1, CV_64FC4);
    ReTotal=0;
    for (int i=0; i<n; i++){
        if (isHold[i]){
            CvScalar temp;
            temp = cvGet1D(pDetections, i);
            cvSet1D(*pModes, ReTotal, temp);
            ReTotal++;
        }
    }
    delete[] isHold;
}

void PedDetector::MixScale_NonMaxSuppression(CvMat *pDetections,CvMat **pModes){
    qsort((void *)pDetections->data.db, pDetections->rows, 4*sizeof(double), cmp_detection_by_score);
    int n = pDetections->rows;
    int ReTotal=0;
    bool *isHold = new bool[n];
    memset(isHold, 1, n*sizeof(bool));
    
    for(int i=0;i<n;i++){
        if (!isHold[i]) continue;
        ReTotal++;
        CvScalar Di;
        Di = cvGet1D(pDetections, i);
        for(int j=i+1;j<n;j++){
            CvScalar Dj;
            Dj = cvGet1D(pDetections, j);
            double overlapRate;
            //计算两个检测框的重合面积比率
            overlapRate=GetOverlapRate(Di.val, Dj.val);
            double dynThre;
            dynThre=DynamicHhreshold(Di.val, Dj.val);
            cout<<"当前阈值为："<<dynThre<<endl;
            if(overlapRate<0) continue;
            if(overlapRate>dynThre) isHold[j] =false;
        }
    }
    *pModes = cvCreateMat(ReTotal, 1, CV_64FC4);
    ReTotal=0;
    for (int i=0; i<n; i++){
        if (isHold[i]){
            CvScalar temp;
            temp = cvGet1D(pDetections, i);
            cvSet1D(*pModes, ReTotal, temp);
            ReTotal++;
        }
    }
    delete[] isHold;
    
}
//采取保留外围窗口策略
void PedDetector::OutWin_NonMaxSuppression(CvMat *pDetections,CvMat **pModes){
    qsort((void *)pDetections->data.db, pDetections->rows, 4*sizeof(double), cmp_detection_by_score);
    int n = pDetections->rows;
    int ReTotal=0;
    bool *isHold = new bool[n];
    memset(isHold, 1, n*sizeof(bool));
    
    for(int i=0;i<n;i++){
        if (!isHold[i]) continue;
        ReTotal++;
        CvScalar Di;
        Di = cvGet1D(pDetections, i);
        for(int j=i+1;j<n;j++){
            CvScalar Dj;
            Dj = cvGet1D(pDetections, j);
            if(SaveOutWin(Di.val, Dj.val)){
                //交换抑制窗口和被抑制窗口的分数
                double temp;
                cout<<"di length:"<<sizeof(Di.val) / sizeof(Di.val[0])<<endl;
                cout<<"dj length:"<<sizeof(Dj.val) / sizeof(Dj.val[0])<<endl;
                cout<<"isHold length: "<<sizeof(isHold)/sizeof(isHold[0])<<" and i:"<<i<<endl;
                temp=Di.val[3];
                Di.val[3]=Dj.val[3];
                Dj.val[3]=temp;
                isHold[i]=false;
            }else{
                
                double overlapRate;
                //计算两个检测框的重合面积比率
                overlapRate=GetOverlapRate(Di.val, Dj.val);
                double dynThre;
                dynThre=DynamicHhreshold(Di.val, Dj.val);
                if(overlapRate<0) continue;
                if(overlapRate>dynThre) isHold[j] = false;
                
            }
        }
    }
    *pModes = cvCreateMat(ReTotal, 1, CV_64FC4);
    ReTotal=0;
    for (int i=0; i<n; i++){
        if (isHold[i]){
            CvScalar temp;
            temp = cvGet1D(pDetections, i);
            cvSet1D(*pModes, ReTotal, temp);
            ReTotal++;
        }
    }
    delete[] isHold;
    
}

float* convConst(float* I,int h, int w,int d, float r)
{
    int s=1;
    float *O = new float[h*w*d];
    if (r<=1){
        convTri1(I, O, h, w, d, (float)12/r/(r+2)-2, s);
    }
    else{
        convTri(I, O, h, w, d, (int)(r+0.1), s);
    }
    return O;
}

bool PedDetector::GetChnFtrs(float *pImgInput, float h0, float w0,int shrink_rate, float nowScale,CvMat *ChnFtrs[])
{
    float *I, *M, *O, *LUV, *S, *H;
    int h,w, wS, hS, d=3;
    int normRad=5;
    float normConst = 0.005;
    
    // 将原始图像数据缩放到当前尺度
    h = shrink_rate*(int)(h0/nowScale/shrink_rate+0.5), w = shrink_rate*(int)(w0/nowScale/shrink_rate+0.5);
    float *data = new float[h*w*3];
    wS = (w/m_Shrink), hS = (h/m_Shrink);
    float_resample(pImgInput, data, w0, w, h0, h, 3, 1.0);
    // 对图像数据进行卷积操作
    I = convConst(data, w, h, d, 1);
    
    //free(luvImg);
    M = new float[w*h];
    O = new float[w*h];
    // 计算梯度幅值
    gradMag(I, M, O, w, h, 3 );
    // 对梯度幅值图像数据进行卷积操作
    S = convConst(M, w, h, 1, normRad);
    // 归一化
    gradMagNorm(M, S, w, h, normConst);
    H = new float[wS*hS*6];
    memset(H, 0, wS*hS*6*sizeof(float));
    // 计算梯度方向
    gradHist(M, O, H, w, h, m_Shrink, 6, false);
    
    float *M_shrink = new float[wS*hS];
    float *I_shrink = new float[wS*hS*3];
    // 对图像进行稀疏重采样操作，相当于计算并列m_Shrink*m_Shrink矩形框特征
    float_resample(M, M_shrink, w, wS, h, hS, 1, (float)1.0);
    float_resample(I, I_shrink, w, wS, h, hS, 3, (float)1.0);
    
    // 保存最终结果
    for (int i=0; i<3; i++){
        ChnFtrs[i] = cvCreateMat(hS, wS, CV_32FC1);
        for (int j=0; j<hS*wS; j++){
            ChnFtrs[i]->data.fl[j] = I_shrink[i*hS*wS+j];
        }
    }
    ChnFtrs[3] = cvCreateMat(hS, wS, CV_32FC1);
    for (int i=0; i<hS*wS; i++){
        ChnFtrs[3]->data.fl[i] = M_shrink[i];
    }
    for (int i=4; i<10; i++){
        ChnFtrs[i] = cvCreateMat(hS, wS, CV_32FC1);
        for (int j=0; j<hS*wS; j++){
            int mod_i = (13-i)%6; // H的输出数据排列方向相反，需反向处理。
            ChnFtrs[i]->data.fl[j] = H[mod_i*hS*wS+j];
        }
    }
    
    free(I);
    free(M);
    free(O);
    free(S);
    free(M_shrink);
    free(I_shrink);
    free(H);
    free(data);
    
    return true;
}
