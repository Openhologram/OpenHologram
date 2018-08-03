// Copyright (C) Tomoyoshi Shimobaba 2011-

#include "cwo.h"
#include"cwoIteOptmz.h"


int cwoIteOptmz::GS(
	CWO *src, CWO *src_rnd, CWO *dst,
	float z, 
	float ox, float oy, 
	float s_px, float s_py, 
	float d_px, float d_py,
	int ite_num, float wl, int diff_type)
{
//	if(src->GetFieldType()==CWO_FLD_COMPLEX) return CWO_ERROR;

	src_rnd->SetWaveLength(wl);
	src_rnd->SetSrcPitch(s_px,s_py);
	src_rnd->SetDstPitch(d_px,d_py);
	src_rnd->SetSrcOffset(ox,oy);
	src_rnd->Diffract(z,diff_type);
	src_rnd->Phase();

	for(int i=0;i<ite_num;i++){
		//Reconstruct
		src_rnd->Cplx();
		src_rnd->SetSrcOffset(-ox,-oy);
		src_rnd->SetSrcPitch(d_px,d_py);
		src_rnd->SetDstPitch(s_px,s_py);
		src_rnd->Diffract(-z,diff_type);

		//constrait
		src_rnd->Phase();
		src_rnd->Cplx((*src),(*src_rnd));
	
		//make CGH
		src_rnd->SetSrcOffset(ox,oy);	
		src_rnd->SetSrcPitch(s_px,s_py);
		src_rnd->SetDstPitch(d_px,d_py);
		src_rnd->Diffract(z,diff_type);
		src_rnd->Phase();
	
	}

	src_rnd->Cplx();
	
	(*dst)=(*src_rnd);

	return CWO_SUCCESS;
}


