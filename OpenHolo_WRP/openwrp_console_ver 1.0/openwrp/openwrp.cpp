// openwrp.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"

#include "WRP.h"

#pragma comment(lib, "cwo.lib")


int _tmain(int argc, _TCHAR* argv[])
{
	OPHWRP oh_wrp;

	oh_wrp.readxyz("object.txt");   //read point cloud
	oh_wrp.Setparameter(532e-9, 8.0e-6, 515);  //read parameter
	oh_wrp.SingleWRP(0.5e-3);
	oh_wrp.SavetoImage("wrp_table.bmp");
	oh_wrp.Fresnelpropagation(0.05);
	oh_wrp.SavetoImage("wrp_result.bmp");

}