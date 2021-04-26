#include<opencv2/opencv.hpp>
#include<stdarg.h>
using namespace cv;
const double PI = 3.1415926;
#define CV_SORT_EVERY_ROW    0
#define CV_SORT_EVERY_COLUMN 1
#define CV_SORT_ASCENDING    0
#define CV_SORT_DESCENDING   16



Mat conv2D(cv:: Mat img,int kSize,cv:: Mat kernel) {
    int row = img.rows;
    int col = img.cols;
    cv::Rect rect;
    Mat src;
    src = img.clone();
    printf("kernel类型：%d，img类型:%d\n",kernel.type(),src.type());
   
    Mat filterImg = Mat::zeros(row, col, CV_64FC1);
    int gaussCenter = kSize / 2;
    for (int i = gaussCenter; i < row - gaussCenter; i++) {
        for (int j = gaussCenter; j < col - gaussCenter; j++) {
            rect.x = j - gaussCenter;
            rect.y = i - gaussCenter;
            rect.width = kSize;
            rect.height = kSize;
            //printf("******************\n");
            printf("%d %d\n", i, j);
           
            Mat temp = src(rect);
           
            filterImg.at<double>(i, j) = cv::sum(kernel.mul(src(rect))).val[0];
            
        }
    }
    return filterImg;
}

void calGrad(cv:: Mat img,Mat* whereGrad1,Mat* grad1,Mat* thetas1) {
    int row = img.rows;
    int col = img.cols;
    cv::Rect temp;
    Mat gradX = Mat::zeros(row, col, CV_64FC1); //水平梯度
    Mat gradY = Mat::zeros(row, col, CV_64FC1); //垂直梯度
     
    Mat grad = *grad1;//梯度幅值
    
    Mat thetas = *thetas1;//梯度角度
    
    Mat whereGrad = *whereGrad1;//标记不同区域
     //x方向的Sobel算子
    Mat Sx = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    //y方向的Sobel算子
    Mat Sy = (cv::Mat_<double >(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    for (int i = 1; i < row - 1; i++) {
        for (int j = 1; j < col - 1; j++) {
            temp.x = j - 1;
            temp.y = i - 1;
            temp.width = 3;
            temp.height = 3;
            Mat rectImg = Mat::zeros(3, 3, CV_64FC1);
            img(temp).copyTo(rectImg);
            //梯度和角度
            gradX.at<double>(i, j) += cv::sum(rectImg.mul(Sx)).val[0];
            gradY.at<double>(i, j) += cv::sum(rectImg.mul(Sy)).val[0];
            grad.at<double>(i, j) = sqrt(pow(gradX.at<double>(i, j), 2) + pow(gradY.at<double>(i, j), 2));
            thetas.at<double>(i, j) = atan(gradY.at<double>(i, j) / gradX.at<double>(i, j));
            if (0 <= thetas.at<double>(i, j) <= (PI / 4.0)) {
                whereGrad.at<double>(i, j) = 0;
            }
            else if (PI / 4.0 < thetas.at<double>(i, j) <= (PI / 2.0)) {
                whereGrad.at<double>(i, j) = 1;
            }
            else if (-PI / 2.0 <= thetas.at<double>(i, j) <= (-PI / 4.0)) {
                whereGrad.at<double>(i, j) = 2;
            }
            else if (-PI / 4.0 < thetas.at<double>(i, j) < 0) {
                whereGrad.at<double>(i, j) = 3;
            }
        }
    }
    double gradMax;
    cv::minMaxLoc(grad, &gradMax);
    if (gradMax != 0) {
        grad = grad / gradMax;
    }
    *grad1 = grad;
    *whereGrad1 = whereGrad;
    *thetas1 = thetas;

}

Mat NMS(cv:: Mat img, Mat grad, Mat whereGrad,Mat thetas,double highThres,double lowThres) {
    int row = img.rows;
    int col = img.cols;
    Mat result = Mat::zeros(row, col, CV_64FC1);
    cv::Mat caculateValue = cv::Mat::zeros(row, col, CV_64FC1); //grad变成一维
    resize(grad, caculateValue, Size(1, grad.rows * grad.cols));
    cv::sort(caculateValue, caculateValue, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);//升序
    long long highIndex = row * col * highThres;
    double highValue = caculateValue.at<double>(highIndex, 0); //最大阈值
    double lowValue = highValue * lowThres;
    //NMS
    for (int i = 1; i < row - 1; i++) {
        for (int j = 1; j < col - 1; j++) {
            // 八个方位
            double N = grad.at<double>(i - 1, j);
            double NE = grad.at<double>(i - 1, j + 1);
            double E = grad.at<double>(i, j + 1);
            double SE = grad.at<double>(i + 1, j + 1);
            double S = grad.at<double>(i + 1, j);
            double SW = grad.at<double>(i - 1, j - 1);
            double W = grad.at<double>(i, j - 1);
            double NW = grad.at<double>(i - 1, j - 1); // 区域判断，线性插值处理
            double tanThead; // tan角度
            double Gp1; // 两个方向的梯度强度
            double Gp2; // 求角度，绝对值
            tanThead = abs(tan(thetas.at<double>(i, j)));
            //线性插值
            switch ((int)whereGrad.at<double>(i, j)) {
            case 0: Gp1 = (1 - tanThead) * E + tanThead * NE; Gp2 = (1 - tanThead) * W + tanThead * SW; break;
            case 1: Gp1 = (1 - tanThead) * N + tanThead * NE; Gp2 = (1 - tanThead) * S + tanThead * SW; break;
            case 2: Gp1 = (1 - tanThead) * N + tanThead * NW; Gp2 = (1 - tanThead) * S + tanThead * SE; break;
            case 3: Gp1 = (1 - tanThead) * W + tanThead * NW; Gp2 = (1 - tanThead) * E + tanThead * SE; break;
            default: break;
            }
            // NMS -非极大值抑制和双阈值检测
            if (grad.at<double>(i, j) >= Gp1 && grad.at<double>(i, j) >= Gp2) {
                //双阈值检测
                if (grad.at<double>(i, j) >= highValue) {
                    grad.at<double>(i, j) = highValue;
                    result.at<double>(i, j) = 255;
                }
                else if (grad.at<double>(i, j) < lowValue) {
                    grad.at<double>(i, j) = 0;
                }
                else {
                    grad.at<double>(i, j) = lowValue;
                }
            }
            else {
                grad.at<double>(i, j) = 0;
            }
        }
    }
    //抑制孤立低阈值点 3*3. 找到高阈值就255
    cv::Rect temp;
    for (int i = 1; i < row - 1; i++) {
        for (int j = 1; j < col - 1; j++) {
            if (grad.at<double>(i, j) == lowValue) {
                //3*3 区域强度
                temp.x = i - 1;
                temp.y = j - 1;
                temp.width = 3;
                temp.height = 3;
                for (int x = 0; x < 3; x++) {
                    for (int y = 0; y < 3; y++) {
                        if (grad(temp).at<double>(x, y) == highValue) {
                            result.at<double>(i, j) = 255;
                            break;
                        }
                    }
                }
            }
        }
    }
    return result;
}


Mat getKernel(int kSize) {
    int gaussCenter = kSize / 2;
    double  sigma = 1;
    Mat guassKernel = Mat::zeros(kSize, kSize, CV_64FC1);
    for (int i = 0; i < kSize; i++) {
        for (int j = 0; j < kSize; j++) {
            guassKernel.at<double>(i, j) = (1.0 / (2.0 * PI * sigma * sigma)) * (double)exp(-(((double)pow((i - (gaussCenter + 1)), 2) + (double)pow((j - (gaussCenter + 1)), 2)) / (2.0 * sigma * sigma)));
        }
    }
    Scalar sumValueScalar = cv::sum(guassKernel);
    double sum = sumValueScalar.val[0];
    guassKernel = guassKernel / sum;
    return guassKernel;
}



int main()
{
    Mat girl = imread("lena512color.tiff"); //载入图像到Mat
    
    cvtColor(girl, girl, COLOR_BGR2GRAY);
    int co = girl.cols;
    int ro = girl.rows;
    int ch = girl.channels();
    printf("%d %d %d", co, ro, ch);
    imshow("女孩头像", girl);//显示名为 "【1】动漫图"的窗口  
    waitKey(0);
    Mat gussKernel = getKernel(3);
    printf("getkernel***********\n");
    girl.convertTo(girl, 6);
    Mat filterImg = conv2D(girl, 3, gussKernel);
    printf("conv2d*********\n");
    Mat grad = Mat::zeros(ro, co, CV_64FC1); //梯度幅值
    Mat thetas = Mat::zeros(ro, co, CV_64FC1); //梯度角度
    Mat whereGrad = Mat::zeros(ro, co, CV_64FC1);//区域
    calGrad(filterImg, &whereGrad, &grad, &thetas);
    printf("calGrad*********\n");
    Mat results = NMS(girl, grad, whereGrad, thetas, 0.9, 0.05);
    imshow("边缘检测结果", results);
    waitKey(0);
    imwrite("results3.bmp", results);


    return 0;
}