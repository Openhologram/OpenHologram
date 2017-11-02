function [ phase_H_image ] = Hologram_WRP( obj, Hologram_resolution, SLM_pitch )
%generate hologram by wavefront recording plane
%obj is point cloud based data(x,y,z), ASC, txt etc. 
%kinect depth and color information modeling

%%input 
obj=load('Demo Head - Point Cloud.ASC');

%% computer generated hologram-WRP
%%%% WRP

lambda = 532e-9;
k = 2*pi/lambda;

obj_center=(max(obj)+min(obj))./2;
obj(:,1)= (obj(:,1)-obj_center(1))/15000;
obj(:,2)= (obj(:,2)-obj_center(2))/15000;
obj(:,3) = (obj(:,3)-obj_center(3))/40000;


size_all = 4e-3;  %hologram_size

size_obj = length(obj(:,1));

Hologram_wrp = zeros(Hologram_resolution);
Hologram_sampling_interval = size_all/Hologram_resolution;

z_wrp = 0.5e-3;  %object to wrp

z=z_wrp-obj(:,3);
N=round(abs(lambda.*z./(Hologram_sampling_interval^2)/2)+0.5).*2-1;  %obj_subhologram_N
Nx = round(X./Hologram_sampling_interval)+(Hologram_resolution-1)/2;
Ny = round(Y./Hologram_sampling_interval)+(Hologram_resolution-1)/2;

for o = 1: size_obj

    [y_run, x_run]= meshgrid((-(N(o)-1)/2:(N(o)-1)/2)*Hologram_sampling_interval,(-(N(o)-1)/2:(N(o)-1)/2)*Hologram_sampling_interval);
    r = sign(z(o))*sqrt(z(o)^2 + y_run.^2 + x_run.^2);
    Sub_hologram = exp(1j*rand*2*pi)*exp(1j*k*r)./r;     
    temp=zeros(Hologram_resolution+25, Hologram_resolution+25);    
    temp(Nx(o):Nx(o)+N(o)-1,Ny(o):Ny(o)+N(o)-1)= Sub_hologram;
    Hologram_wrp=Hologram_wrp+temp((N(o)+1)/2:Hologram_resolution+(N(o)-1)/2,(N(o)+1)/2:Hologram_resolution+(N(o)-1)/2);  

end

WRPHologram = WRP_FresnelPropagation(Hologram_wrp, dx, dy, d, lambda);

phaseadd =WRPHologram;
phase_H = angle(phaseadd) + pi;
phase_H_image = uint8(255*phase_H/max(max(phase_H)));
imwrite(phase_H_image, 'Orth_wrp.bmp', 'bmp');

end

