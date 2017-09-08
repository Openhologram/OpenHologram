function [ phase_H_image ] = Hologram_WRP( obj, hologram_resolution, SLM_pitch )
%generate hologram by wavefront recording plane
%obj is point cloud based data(x,y,z), ASC, txt etc. 
%kinect depth and color information modeling

sipha=90;
betha=90;
gamma=90;
XRS=[1 0 0;
     0 cosd(sipha) -sind(sipha);
     0 sind(sipha) cosd(sipha);];
YRS=[cosd(-betha) 0 -sind(-betha) ;
    0 1 0 ;
    sind(-betha) 0 cosd(-betha) ;];
ZRS=[cosd(gamma) -sind(gamma) 0 ;
    sind(gamma) cosd(gamma) 0 ;
    0 0 1 ;]; 
obj=obj*XRS*YRS*ZRS;
points_show(obj);

%% 2. create direction view 
%%% cartesian to Spherical coordinates [azimuth,elevation,r] = cart2sph(x,y,z)

obj_center=(max(obj)+min(obj))./2;
vw=[obj_center(1),obj_center(2),obj_center(3)+100];

obj=obj(HPR(obj,vw,4),:);

depthmap=pointcloud2image(obj(:,1),obj(:,2),obj(:,3),500,500);
J = imcomplement(depthmap)*255;
figure,imshow(J,[]);
points_show(obj);

%% 3. computer generated hologram-WRP
%%%% WRP

lambda = 532e-9;
k = 2*pi/lambda;

obj(:,1)= (obj(:,1)-obj_center(1))/15000;
obj(:,2)= (obj(:,2)-obj_center(2))/15000;
obj(:,3) = (obj(:,3)-obj_center(3))/40000;

X = obj(:,1);
Y = obj(:,2);
Z = obj(:,3);

t=hologram_resolution;
min_depth = min(Z);
max_depth = max(Z);
size_all = 4e-3;  %hologram_size
figure; plot3(X,Y,Z,'.');xlabel ('x');ylabel ('y');zlabel ('z')

[size_obj,a] = size(obj);
Hologram_size = t;
tic
Hologram_wrp = zeros(Hologram_size);
Hologram_sampling_interval = size_all/Hologram_size;

z_wrp = 0.5e-3;  %object to wrp
z0 = Hologram_sampling_interval*(size_all*2)/lambda;

%GPU
tic
z=z_wrp-Z;
N=round(abs(lambda.*z./(Hologram_sampling_interval^2)/2)+0.5).*2-1;  %obj_subhologram_N
Nx = round(X./Hologram_sampling_interval)+(Hologram_size-1)/2;
Ny = round(Y./Hologram_sampling_interval)+(Hologram_size-1)/2;

for o = 1: size_obj

    [y_run, x_run]= meshgrid((-(N(o)-1)/2:(N(o)-1)/2)*Hologram_sampling_interval,(-(N(o)-1)/2:(N(o)-1)/2)*Hologram_sampling_interval);
    r = sign(z(o))*sqrt(z(o)^2 + y_run.^2 + x_run.^2);
    Sub_hologram = exp(1j*rand*2*pi)*exp(1j*k*r)./r;     
    temp=zeros(Hologram_size+25, Hologram_size+25);    
    temp(Nx(o):Nx(o)+N(o)-1,Ny(o):Ny(o)+N(o)-1)= Sub_hologram;
    Hologram_wrp=Hologram_wrp+temp((N(o)+1)/2:Hologram_size+(N(o)-1)/2,(N(o)+1)/2:Hologram_size+(N(o)-1)/2);  

end

Hologram_wrp=gather(Hologram_wrp);
toc

WRPHologram = FresnelPropagation(Hologram_wrp, dx, dy, d, lambda);

phaseadd =WRPHologram;
phase_H = angle(phaseadd) + pi;
phase_H_image = uint8(255*phase_H/max(max(phase_H)));
imwrite(phase_H_image, 'Orth_wrp.bmp', 'bmp');

end

