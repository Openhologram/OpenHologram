%% Parameters
lambda=532e-9;
dx = 2.7e-6;  
dy = 2.7e-6;

%% Load phase shifted holograms
I0 = double(rgb2gray(imread('./samplePhaseShiftedHolograms/0930_005.bmp')));
I90 = double(rgb2gray(imread('./samplePhaseShiftedHolograms/0930_006.bmp')));
I180 = double(rgb2gray(imread('./samplePhaseShiftedHolograms/0930_007.bmp')));
I270 = double(rgb2gray(imread('./samplePhaseShiftedHolograms/0930_008.bmp')));

%% Get complex field
complexField = fn_getComplexFieldFrom4PhaseShiftedHolograms(I0,I90,I180,I270);

%% Show amplitude and phase of the complex field
figure(); imagesc(abs(complexField)); colorbar(); axis equal; title('amplitude of the complex field')
figure(); imagesc(angle(complexField)); colorbar(); axis equal; title('phase of the complex field')

%% Numerical propagation
z = 57e-2;
gpuAvailability=false;

[numericalReconstruction,du,dv] = fn_FresnelPropagation_as(complexField, dx, dy, z, lambda, false);
figure(); imagesc(abs(numericalReconstruction)); axis equal; title(['numerical reconstruction at z=',num2str(z),'m'])
