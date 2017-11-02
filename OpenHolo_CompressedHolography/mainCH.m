% Example code using compressed holography functions
clear

%% Input field
load('./sample_complex_field/sampleComplexField')
complexField = complexField/max(abs(complexField(:)));

%% Parameter Setting
zc = 0.058;         % central distance of the object
zmin = zc-4e-3;     % minimum distance
zmax = zc+4e-3;     % maximum distance
dz = 0.8e-3;

[ny, nx] = size(complexField);
nz=10;              % Number of the depth planes for reconstruction
z = (0:(nz-1))*dz + zmin;   % depth planes

dx=20e-6; dy=20e-6; % sampling pitch of the complexField data 
lambda=532e-8;      % wavelength of the complexField data

%% Visualize complex field
figure();imagesc(abs(complexField));title('amplitude of the complex field');axis image; axis off; colormap(hot); colorbar;
figure();imagesc(angle(complexField));title('phase of the complex field');axis image; axis off; colormap(hot); colorbar;

%% Simple numerical propagations
f_nb = fn_2Dto3D(fn_ri2c(fn_plane2volume(complexField, dx, dy, -z, lambda)), nz);
figure;imagesc(fn_showcube(abs(f_nb), 5));title('Reconstructions by simple numerical propagation');axis image;drawnow; axis off; colormap(hot); colorbar;

%% Compressed holography
maxIter=100;
f_ch = fn_compressedHolography(complexField, dx, dy, lambda, z, maxIter);
figure;imagesc(fn_showcube(abs(f_ch), 5));title('Reconstructions by Compressive holography');axis image;drawnow; axis off; colormap(hot); colorbar;





