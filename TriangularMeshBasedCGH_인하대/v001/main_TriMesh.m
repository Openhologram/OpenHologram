% Triangular Mesh
% Object -> Hologram plane -> Reconstruction

tic
clc; clear;

%% Input

% 01 Hologram parameters
% 02 Shading parameters
% 03 Load object & adjust size
% 04 Output file name

%% Misc.
% if you don't want images to pop up, put " fig=0 ", or put " fig=1 ".
% if you don't want images to be saved, put " sav=0 ", or put " sav=1 ".
fig=1;  sav=1;
gpuAvailable=1; % 1: when gpu is available,  0 when not

%% 01 Hologram parameters

holoParam.wavelength=532e-9;    %%%%% LASER wavelength in meter

holoParam.Nx=1920;     %%%%% SLM resolution ( holo.Nx * holo.Ny )
holoParam.Ny=1080; 

holoParam.dx=8e-6;   %%%%% SLM pixel pitch in meter
holoParam.dy=8e-6;

holoParam.sizeX = holoParam.Nx*holoParam.dx;   %%%%% SLM physical length in meter
holoParam.sizeY = holoParam.Ny*holoParam.dy;

%% 02 Shading Effect

% Illumination position / [0 0 0] (if you don't want to put illumination effect)
shadingParam.illu=[1 1 1];

% Continuous Shading
shadingParam.con = 1;    % put or not

%% 03 Load object and adjust size

% DATA STYLE : 1 triangle / 1 row ( N by 9 matrix )
% Each row: [x1,y1,z1,   x2,y2,z2,  x3,y3,z3] --> 3 vertices of a mesh
meshDataFileName = './teapot/mesh_teapot.txt';

% Object shift from center
shiftX=0;   
shiftY=0;  
shiftZ=0.1;
objectSize = holoParam.sizeX/3;

%% 04 Output File Name - Save File

% HOLOGRAM FILE NAME (SAVE) : hologram_name
% HOLOGRAM RECONSTRUCTION FILE NAME (SAVE) : reconst_name

hologramFileName='hologram.jpg';
recFileName='recon.jpg';

%% %%%%%%%%%%%%%%%%%% Setting done %%%%%%%%%%%%%%%%%% %%

%% Object File Load

obj=load(meshDataFileName);
obj = fn_normalizeCenteringObj(obj);
obj = fn_scaleShiftObj(obj, [objectSize, objectSize, objectSize], [shiftX, shiftY, shiftZ]);

%% Hologram generation
hologram = fn_genH(obj, holoParam, shadingParam, gpuAvailable);


%% Numerical reconstruction
[rec,du,dv] = fn_FresnelPropagation_as(hologram, holoParam.dx, holoParam.dy, shiftZ, holoParam. wavelength, gpuAvailable);


%% Figure

if fig==0
else
    
axisx=(-holoParam.Nx/2:holoParam.Nx/2-1)*holoParam.dx;
axisy=(-holoParam.Ny/2:holoParam.Ny/2-1)*holoParam.dy;

figure(1); imagesc(axisx,axisy,abs(hologram)); colormap(gray); colorbar(); title('Hologram amplitude') 
figure(2); imagesc(axisx, axisy, angle(hologram)); colormap(gray); colorbar(); title('Hologram phase')
figure(3); imagesc(axisx,axisy,abs(rec)); colormap(gray); colorbar(); title('Reconstruction amplitude') 

end

%% Save Files

if sav==0
else
    
hologramAbsNormalized255 = 255*abs(hologram)/max(abs(hologram(:)));     % gray scale
recAbsNormalized255 = 255*abs(rec)/max(abs(rec(:)));     % gray scale

if gpuAvailable==1
    hologramAbsNormalized255 = gather(hologramAbsNormalized255);
    recAbsNormalized255 = gather(recAbsNormalized255);
end
imwrite(mat2gray(hologramAbsNormalized255), hologramFileName);       % hologram plane
imwrite( mat2gray(recAbsNormalized255) , recFileName ); % hologram
end

toc