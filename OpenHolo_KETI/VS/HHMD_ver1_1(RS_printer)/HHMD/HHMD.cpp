// HHMD.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "Define.h"
#define  _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>
#include <memory.h>
#include <time.h>

int parseTokens(FILE *filePtr, char **tokens, char **values)
{
	int i = 0;

	char line[128];

	while (fgets(line, 127, filePtr)) {
		tokens[i] = (char*)malloc(sizeof(char)*64);
		values[i] = (char*)malloc(sizeof(char)*64);

		sscanf(line, "%s = %s", tokens[i], values[i]);

		i++;
	}

	return i;
}

int main(int argc, char* argv[])
{
	char *tokens[20];
	char *values[20];
	char dummy[512];

	FILE *s_fp_pcFile;		// Pointer for point cloud data file
	FILE *s_fp_configFile;	// Pointer for Config file
	PCLOUD *s_p_pcloud;		// Pointer for array of point cloud
	SPEC s_s_specs;
	int s_i_pointNum = 0;	// Total number of points in point cloud

	int temp = 0;


	if(argc < 4){
		LOG("Usage: PointCloudFIle ConfigFile DestinationFile. Press \"CTRL+c\" to stop.");	
		return 0;
	}
	// Opening point cloud data file
	if(fopen_s(&s_fp_pcFile, (char*)argv[1],"r")) {	
		LOG("Cannot read point cloud file. Press \"CTRL+c\" to stop.");	
		return 0;
	}
	// Opening configuration file
	if(fopen_s(&s_fp_configFile, (char*)argv[2],"r")) {	
		LOG("Cannot read config file. Press \"CTRL+c\" to stop.");	
		return 0;
	}
	// Starting parser for point cloud data file
	if(fscanf_s(s_fp_pcFile, "%d", &s_i_pointNum) <= 0) {	
		LOG("Cannot read total number of points. Press \"CTRL+c\" to stop.");	
		return 0;
	}
	// Preparing buffer to read point cloud data
	s_p_pcloud = (PCLOUD*)malloc(sizeof(PCLOUD)*s_i_pointNum); 
	memset(s_p_pcloud, 0x00, sizeof(PCLOUD)*s_i_pointNum);

	// Parsing (index, x,y,z, amplitude, phase) from point cloud data file
	for(int i= 0; i < s_i_pointNum ; i++) {
		if(fscanf_s(s_fp_pcFile, "%d", &((s_p_pcloud+i)->index)) <= 0)	break; 
		if(fscanf_s(s_fp_pcFile, "%f", &((s_p_pcloud+i)->x)) <= 0)		break;  
		if(fscanf_s(s_fp_pcFile, "%f", &((s_p_pcloud+i)->y)) <= 0)		break;  
		if(fscanf_s(s_fp_pcFile, "%f", &((s_p_pcloud+i)->z)) <= 0)		break;  
		if(fscanf_s(s_fp_pcFile, "%f", &((s_p_pcloud+i)->phase)) <= 0)	break;  
		if(fscanf_s(s_fp_pcFile, "%f", &((s_p_pcloud+i)->amp)) <= 0)	break; 
	}

	// Starting parser for configuration file
	int count = parseTokens(s_fp_configFile, tokens, values);
	
	if(count != 17){
		LOG("Check the number of specifications. Press \"CTRL+c\" to stop.");	
		return 0;
	}

	temp = 0;
	s_s_specs.pcScaleX = atof(values[temp]); temp++;
	s_s_specs.pcScaleY = atof(values[temp]); temp++;
	s_s_specs.pcScaleZ = atof(values[temp]); temp++;
	s_s_specs.offsetDepth = atof(values[temp]); temp++;
	s_s_specs.sPitchX = atof(values[temp]); temp++;
	s_s_specs.sPitchY = atof(values[temp]); temp++;
	s_s_specs.sNumX = atoi(values[temp]); temp++;
	s_s_specs.sNumY = atoi(values[temp]); temp++;
	s_s_specs.filterShape = values[temp]; temp++;
	s_s_specs.wFilterX = atof(values[temp]); temp++;
	s_s_specs.wFilterY = atof(values[temp]); temp++;
	s_s_specs.fIn = atof(values[temp]); temp++;
	s_s_specs.fOut = atof(values[temp]); temp++;
	s_s_specs.fEye = atof(values[temp]); temp++;
	s_s_specs.lambda = atof(values[temp]); temp++;
	s_s_specs.tiltAngleX = atof(values[temp]); temp++;
	s_s_specs.tiltAngleY = atof(values[temp]); temp++;

	float f_1 = s_s_specs.fIn;
	float f_2 = s_s_specs.fOut;
//	float f_3 = s_s_specs.fEye;
	int N_x = s_s_specs.sNumX;     // 852
	int N_y = s_s_specs.sNumY;     //852
	float tiltX = M_PI/180*s_s_specs.tiltAngleX;
	float tiltY = M_PI/180*s_s_specs.tiltAngleY;

	float k = 2*M_PI/s_s_specs.lambda; // Wavenumber

	// Dealing with information of bandpass filter
/*
	int flagCirc = 1;
	if (strcmp(s_s_specs.filterShape, "circle") != 0){
		flagCirc = 0;
	}
	float w_FX, w_FY;
*/
	// Calculating tilt angle in viewing window domain
	float theta_x = tiltX;
	float theta_y = tiltY;

	// Pixel pitch at eyepiece lens plane (by simple magnification) ==> SLM pitch
	float p_x = s_s_specs.sPitchX; 
	float p_y = s_s_specs.sPitchY;

	// Width (Length) of complex field at eyepiece plane (by simple magnification)
	float L_x = p_x*N_x; 
	float L_y = p_y*N_y;

	// Buffer for fringe pattern
	float *u_h = (float*)malloc(sizeof(float)*N_x*N_y);
	memset(u_h, 0, sizeof(float)*N_x*N_y);
	unsigned char *qnt_u = (unsigned char*)malloc(sizeof(unsigned char)*N_x*N_y);
	memset(qnt_u, 0x00, sizeof(unsigned char)*N_x*N_y);
	char *flag_addressed = (char*)malloc(sizeof(char)*N_x*N_y);
	memset(flag_addressed, 0x00, sizeof(char)*N_x*N_y);

	LOG("Bitmap FIle Generation start");
	int n;
		#pragma omp parallel for private(n)

	// please check this
	for(n = 0; n < s_i_pointNum; n++){
		
		float x_pc = (s_p_pcloud + n)->x * (s_s_specs.pcScaleX);   // x_pc(s_x) : 3d object coord
		float y_pc = (s_p_pcloud + n)->y * (s_s_specs.pcScaleY);
		float z_pc = (s_p_pcloud + n)->z * (s_s_specs.pcScaleZ) + s_s_specs.offsetDepth;
		float amp_pc = (s_p_pcloud + n)->amp;
		float phase_pc = 2*M_PI*(s_p_pcloud + n)->phase; // Phase in data file is assumed to have a value between 0 to 1.


			for (int itr_y = 0; itr_y < N_y; itr_y++){
				// Y coordinate of the current pixel
				float temp_y_cord = L_y/2 - ((float)itr_y + 0.5)*p_y;	// Note that y index is reversed order.
				for (int itr_x = 0; itr_x < N_x; itr_x++){
					// X coordinate of the current pixel
						
						float temp_x_cord = ((float)itr_x + 0.5)*p_x - L_x/2;    // p_x : pixel pitch,  temp_x_cord : SLM coord , x_pc: point cloud x coord 
						float *pixel_pos = u_h + itr_x + itr_y*N_x;
//						char *flag_pos = flag_addressed + itr_x + itr_y*N_x;
						float r = sqrt((temp_x_cord - x_pc)*(temp_x_cord - x_pc) + (temp_y_cord - y_pc)*(temp_y_cord - y_pc) + z_pc*z_pc); // Distance
//						float phi = k*(z_pc/abs(z_pc)*r - temp_x_cord*sin(theta_x) - temp_y_cord*sin(theta_y)); // Phase for printer
						float phi = k*r - k*temp_x_cord*sin(theta_x) - k*temp_y_cord*sin(theta_y); // Phase for printer
						//float calcVal = amp_pc*z_pc/(r*r)*cos(phi + phase_pc);
						float calcVal = amp_pc*cos(phi);
//						float calcVal = amp_pc*cos(phi + phase_pc);
						*(pixel_pos) = *(pixel_pos) + calcVal;
//						*(flag_pos) = 1;
				}
			}
	}
	LOG("Bitmap FIle normalization start");
	// Normalize calculated fringe pattern
	float minVal, maxVal;
	for(int ydx = 0; ydx < N_y; ydx++){
		for(int xdx = 0; xdx < N_x; xdx++){
			float *temp_pos = u_h + xdx + ydx*N_x;
			if((xdx == 0) && (ydx == 0)){
				minVal = *(temp_pos);
				maxVal = *(temp_pos);
			}
			else{
				if(*(temp_pos) < minVal){
					minVal = *(temp_pos);
				}
				if(*(temp_pos) > maxVal){
					maxVal = *(temp_pos);
				}
			}
		}
	}

	for(int ydx = 0; ydx < N_y; ydx++){
		for(int xdx = 0; xdx < N_x; xdx++){
			float *src_pos = u_h + xdx + ydx*N_x;
			unsigned char *res_pos = qnt_u + xdx + (N_y-ydx-1)*N_x;	// Flip image vertically to consider flipping by Fourier transform and projection geometry
//			char *res_flag_pos = flag_addressed + xdx + ydx*N_x;
			
			*(res_pos) = (unsigned char)(((*(src_pos)-minVal)/(maxVal - minVal))*255 + 0.5 );
		
		}
	}
	// bmp fringe pattern file generation
	creatBitmapFile(qnt_u, N_x, N_y, (char*)argv[3]);
	LOG("Bitmap FIle Created");


	fclose(s_fp_pcFile);
	fclose(s_fp_configFile);
	free(s_p_pcloud);
	free(u_h);
	free(qnt_u);

//MPI_END:
	
	return 0;
}



// UTility
// only for debug
void LOG(char* logmsg)
{
	FILE *fp_log;
    char yymmdd[20],hhmmss[20];
	
    get_yymmdd_y2(yymmdd);
    get_hhmmss(hhmmss);

	// log to console
	printf("|%s|%s %s\n",yymmdd,hhmmss,logmsg);
	fflush(NULL);

	char lfname[120];
	sprintf(lfname,"log.txt");

	// log to logfile
	if(!fopen_s(&fp_log, lfname, "a+" ))
    {
		fprintf(fp_log,"|%s|%s %s\n",yymmdd,hhmmss, logmsg);
		fflush(fp_log);
		fclose(fp_log);
    }
}

char* getTimeString()
{
	static char timeStr[32];
	char yymmdd[20],hhmmss[20];
	
    get_yymmdd_y2(yymmdd);
    get_hhmmss(hhmmss);

	sprintf(timeStr,"%s %s",yymmdd, hhmmss);

	return timeStr;

}
void get_yymmdd_y2(char buf[])
{
	struct tm convtime;
	time_t now;
	
	time(&now);
	localtime_s(&convtime, &now);
	sprintf(buf,"%04d-%02d-%02d",
        convtime.tm_year+1900,convtime.tm_mon+1,convtime.tm_mday);
	return ;
}

void get_hhmmss(char buf[])
{
	struct tm convtime;
	time_t now;
	
	time(&now);
	localtime_s(&convtime, &now);
	sprintf(buf,"%02d:%02d:%02d",
        convtime.tm_hour,convtime.tm_min,convtime.tm_sec);
	return;
}