%Main program of hologram generation by wavefront recording plane method
%by Openholo library project
%2017-10-30 update
%
clc;clear;close all;
%% Input the object and prameter

load obj;
lambda = 532e-9;                                % Wave length
k = 2*pi/lambda;       
Hologram_resolution=1025;                       % Hologram resolution     
Hologram_sampling_interval = 3.9e-6;            % Hologram sampling interval


%% setting WRP
z_wrp = 0.5e-3;  %  WRP location
z=z_wrp-obj(:,3);

N=round(abs(lambda.*z./(Hologram_sampling_interval^2)/2)+0.5).*2-1;        %sampling size of N
Nx = round(obj(:,1)./Hologram_sampling_interval)+(Hologram_resolution-1)/2;  
Ny = round(obj(:,2)./Hologram_sampling_interval)+(Hologram_resolution-1)/2;

size_obj = length(obj(:,1));
Hologram_wrp = zeros(Hologram_resolution);

for o = 1: size_obj
    
    fprintf('%d\n',o);  
    [y_run, x_run]= meshgrid((-(N(o)-1)/2:(N(o)-1)/2)*Hologram_sampling_interval,(-(N(o)-1)/2:(N(o)-1)/2)*Hologram_sampling_interval);
    r = sign(z(o))*sqrt(z(o)^2 + y_run.^2 + x_run.^2);
    Sub_hologram = exp(1j*rand*2*pi)*exp(1j*k*r)./r;   
    
    temp=zeros(Hologram_resolution+N(o), Hologram_resolution+N(o));    
    temp(Nx(o):Nx(o)+N(o)-1,Ny(o):Ny(o)+N(o)-1)= Sub_hologram;
    Hologram_wrp=Hologram_wrp+temp((N(o)+1)/2:Hologram_resolution+(N(o)-1)/2,(N(o)+1)/2:Hologram_resolution+(N(o)-1)/2);  
%     figure,imshow(Hologram_wrp)
 
end

%% Fresnel Propagation

ROWS= Hologram_resolution;                                     
COLS= Hologram_resolution;
v=Hologram_sampling_interval.*(ones(COLS,1)*(-ROWS/2:ROWS/2-1))';
h=Hologram_sampling_interval.*(ones(ROWS,1)*(-COLS/2:COLS/2-1));
d=0.05;
WRPHologram = FresnelPropagation(Hologram_wrp, Hologram_sampling_interval, Hologram_sampling_interval, d, lambda);

%% reconstruction

for o=0:1:20  % reconstructed  
    d2 = d+0.002 - o*0.0002;
    original = FresnelPropogation(k,v, h,-d2,WRPHologram);
    figure; imshow(abs(original),[]);
end

phaseadd =WRPHologram;
phase_H = angle(phaseadd) + pi;
phase_H_image = uint8(255*phase_H/max(max(phase_H)));
imwrite(phase_H_image, 'wrp_hologram.bmp', 'bmp');


