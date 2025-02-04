#inlcude "ImgProcess_OpenCV.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

/*
OpenCV codes for edge detection using the Canny method:
strFilePath --- Original image.
*/
void CImgProcess_OpenCV::OpenCV_Canny(CString strFilePath)
{
	// Input image.
	cv::Mat imgMountain = cv::imread(strFilePath);

	// Declare output image and mask image.
	cv::Mat imgOutlines, edgeMask;
	imgOutlines.create(imgMountain.size(), imgMountain.type());

	// Transfer the color image to gray image.
	cv::cvtColor(imgMountain, edgeMask, cv::COLOR_BGR2GRAY);

	// Gaussian smooth.
	cv::blur(edgeMask, edgeMask, cv::Size(5, 5));

	// Canny method (the Sobel operator is imbeded inside).
	cv::Canny(edgeMask, edgeMask, 3, 9, 3);

	// Clear output image.
	imgOutlines = cv::Scalar::all(0);

	// Output edge outlines to imgOutlines.
	imgMountain.copyTo(imgOutlines, edgeMask);

	// Save the processed image as "Edge.png".
	CString saveFile = "Edge.png";
	cv::imwrite(saveFile.GetBuffer(), imgOutlines);
}
