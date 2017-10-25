% CGH from Light field

% This code is an example code for using fn_convertLightField2ComplexField
% In this example, the light field data is assumed to be given as a
% sequence of the image fields each of which has a orthographic view of the
% 3D scene.

% Input to the fn_convertLightField2ComplexField --> matrix(Nv,Nu,Nt,Ns)
% where Nu x Nv : resolution of each perspective image
%       Ns x Nt : # of the perspective images

% Orthographic view images
% Nu x Nv orthographic views (image files) with Ns x Nt resolution are loaded to form LF
% with Nv x Nu x Nt x Ns dimension

%% load orthographic view images
dirName = './sample_orthographic_images/';
fileExt = 'png';
orthographicImageFiles = dir([dirName,'*.',fileExt]);
Nu=20; Nv=20;
Ns=200; Nt=200;

temp = zeros(Nt,Ns,Nv,Nu); 

disp('load all images')
for idxU=1:Nu
    for idxV=1:Nv
        k = (idxV-1)*Nv + idxU;
        temp(:,:,idxV,idxU) = double(rgb2gray(imread([dirName,orthographicImageFiles(k).name])));
    end
end

% temp matrix should be dimension shifted
LF = shiftdim(temp,2); % Now [Nv, Nu, Nt, Ns]

%% Convert the light field to complex field in RS (ray sampling) plane
disp('converting to complex field')
complexField = fn_convertLightField2ComplexField(LF);

%% Propagate the complex field in RS plane to hologram plane
dx=2e-6; dy=2e-6;           % sampling pitch of the RS plane and the hologram plane
lambda = 532e-9;            % wavelength
gpuAvailability = false;

distanceFromRS2H=10e-3;     % distance between the RS plane and the hologram plane
[hologram, dummy1,dummy2] = fn_FresnelPropagation_as(complexField,dx,dy,distanceFromRS2H,lambda,gpuAvailability);
figure(); imagesc(abs(hologram)); axis equal; title('amplitude of the hologram')
figure(); imagesc(angle(hologram)); axis equal; title('phase of the hologram')

%% sample reconstruction
zRec1 = -distanceFromRS2H;
zRec2 = -distanceFromRS2H + 1e-3;

z = zRec1;
[rec, dummy1,dummy2] = fn_FresnelPropagation_as(hologram,dx,dy,z,lambda,gpuAvailability);
figure(); imagesc(abs(rec)); axis equal; title(['reconstruction at z=',num2str(z),'m'])

z = zRec2;
[rec, dummy1,dummy2] = fn_FresnelPropagation_as(hologram,dx,dy,z,lambda,gpuAvailability);
figure(); imagesc(abs(rec)); axis equal; title(['reconstruction at z=',num2str(z),'m'])
