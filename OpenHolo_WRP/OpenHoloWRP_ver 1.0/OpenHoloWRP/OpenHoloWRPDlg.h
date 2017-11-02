
// OpenHoloWRPDlg.h : 헤더 파일
//

#pragma once

#include "cwo.h"
#include "WRP.h"
#pragma comment(lib, "cwo.lib")
// OPHWRPDlg 대화 상자
class OPHWRPDlg : public CDialogEx
{
// 생성입니다.
public:
	OPHWRPDlg(CWnd* pParent = NULL);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
	enum { IDD = IDD_OPENHOLOWRP_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedImgbtn();
	CString m_editstr;
	string m_imgPath;
	OHWRP oh_wrp;

//	void DisplayImage(string filepath);
};
