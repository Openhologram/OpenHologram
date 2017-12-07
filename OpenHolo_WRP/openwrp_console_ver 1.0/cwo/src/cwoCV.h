// Copyright (C) Tomoyoshi Shimobaba 2011-

#ifndef _CWO_CV_H
#define _CWO_CV_H

#include "cwo.h"
#include <opencv2/opencv.hpp>

//!  Class for interface to OpenCV. 
/*!
This class serves the interface between CWO and OpenCV.
*/
class cwoCV {
public:
	//###########################################
	/** @defgroup operator Operators
	*/
	//@{
	cwoCV();
	~cwoCV();
	//@}

	//! Convert CWO object to cv::Mat object
    /*!
	@param src : pointer to CWO object
	@param dst : pointer to cv::Mat object
    */
	void ConvToMat(CWO *src, cv::Mat *dst);
	
	//! Convert cv::Mat object to CWO object
    /*!
	@param src : pointer to cv::Mat object
	@param dst : pointer to CWO object
    */
	void ConvToCWO(cv::Mat *src, CWO *dst);
};

cwoCV::cwoCV()
{
}

cwoCV::~cwoCV()
{
}

void cwoCV::ConvToMat(CWO *src, cv::Mat *dst)
{
	int Nx=src->GetNx();
	int Ny=src->GetNy();
	float* src_p=(float*)src->GetBuffer();

	dst->create(Ny,Nx,CV_32F);
	for( int y = 0 ; y <Ny ; ++y ){
		for( int x = 0 ; x<Nx ; ++x ){
			float pix;
			src->GetPixel(x,y,pix);
			dst->at<float>(y,x) =pix;
		}
	}

}
void cwoCV::ConvToCWO(cv::Mat *src, CWO *dst)
{
	cv::Size size=src->size();
	int Nx=size.width;
	int Ny=size.height;
	
	dst->Create(Nx,Ny);
	dst->SetFieldType(CWO_FLD_INTENSITY);

	for( int y = 0 ; y <Ny ; ++y ){
		for( int x = 0 ; x<Nx ; ++x ){
			float pix;
			dst->SetPixel(x,y,src->at<float>(y,x));
		}
	}

}

#endif
