%clear all
%format short
format long
%%%%%%%%%%% CGH pre-define %%%%%%%%%%%%%%%%%%%%%%%%%

CGH_width = 852;
CGH_height = 852;

rLamda = 0.000000660; 
Red_wavelength = 0.000000660;

Default_depth = -0.06; %1.29; %50cm 0.5;   % 0.06-> -0.06
CGH_scale = 0.05; 
CGH_scale_z = 0.05;

thetaX = 22.5*pi/180;  % radian
thetaY = 22.5*pi/180;  % radian

% WaveNum =9926043.0; % 2pi/lamda
WaveNum =(2*pi)/rLamda;
cal_start = tic;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inFileName ='in2.dat';
outFileName = './RED.bmp';
infp = fopen(inFileName,'r');

m_pointnum = fscanf(infp, '%d', 1);

x = zeros(1,m_pointnum); y = zeros(1,m_pointnum);z = zeros(1,m_pointnum);
%hong = zeros(1,m_pointnum);
amp = zeros(1,m_pointnum); phi = zeros(1,m_pointnum);
total_phase = zeros(CGH_height,CGH_width);  
fringe_total = zeros(CGH_height,CGH_width);


%%%%%%%%%%%%%%%%% Read Point Cloud Data %%%%%%%%%%%%%%%%%%%%%%%%%
for i=1: m_pointnum 
     hong =fscanf(infp, '%d',1);  % hong : number of point cloud -1
     x(i) = fscanf(infp, '%f',1); 
     y(i) =fscanf(infp, '%f',1); 
     z(i) =fscanf(infp, '%f',1); 
     phi(i) =fscanf(infp, '%f',1); 
     amp(i) =fscanf(infp, '%f',1); 
end

%Oscale  =  CGH_scale * (2.0/(N/2.0));
CGH_scaleZ =  CGH_scale;
%Hscale = (rLamda * Default_depth)/(N*Oscale);
Hscale = 0.000000462;    % SLMPixelpitchX,Y
%%%%%%%%%%%%%%%% Fresnel Hologram Generation %%%%%%%%%%%%%%%%%%%%
% s_x,s_y,s_z ==> object 좌표, xl,yl==> hologram fringe pattern 좌표


for no = 1: m_pointnum
   s_x = x(no) * CGH_scale;
   s_y = y(no) * CGH_scale;
%  s_z = z(no) * CGH_scale - Default_depth;
   s_z = z(no) * CGH_scale_z + Default_depth;
    %=> 포인트 계산 순서 표시

            yl_temp = -(( [1:CGH_height] -0.5) - (CGH_height/2) )*Hscale;
            xl_temp = (( [1:CGH_width] -0.5 ) - (CGH_width/2 ) )*Hscale;
            [xl, yl] = meshgrid(xl_temp, yl_temp);
            d =   (yl- s_y).*(yl- s_y) + (xl- s_x).*(xl- s_x) +  s_z*s_z ;
            R =  sqrt(d);

            total_phase = WaveNum*R + WaveNum * xl *sin(thetaX) + WaveNum * yl *sin(thetaY); 
            fringe_total = fringe_total + amp(no)*cos(total_phase);

end

%return
% min/max 
 max_vr = max(fringe_total(:));
 min_vr = min(fringe_total(:));

 % normalization
for yi=1: CGH_height
    for xi = 1: CGH_width
      fringe_total(yi, xi) =int16(  1.0 *  (((fringe_total(yi, xi)-min_vr)/(max_vr - min_vr))*255)); % + 0.5 edit
    end
end

% 256level for bmp file format
fringe_total = uint8(fringe_total);
fringe_total = flipud(fringe_total);   % edit
% bmp file generation
imwrite(fringe_total,outFileName,'bmp');
fclose(infp);

total_time = toc(cal_start);