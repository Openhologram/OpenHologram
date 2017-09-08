% Triangular Mesh
% Object -> Hologram plane -> Reconstruction

tic
clc; clear;

%% HaveToDo

% 01 Condition
% 02 Object shift
% 03 Shading effect
% 04 File name

%% These files will be automatically saved and images will pop up

% hologram plane image file
% hologram reconstruction image file

% if you don't want images to pop up, put " fig=0 ", or put " fig=1 ".
% if you don't want images to be saved, put " sav=0 ", or put " sav=1 ".
fig=1;  sav=1;

%% 01 Condition

w=532*10^-9;    %%%%% LASER wave length

Width=1920;     %%%%% SLM resolution ( Width * Height )
Height=1080; 

pit=8e-6;   %%%%% SLM pixel pitch

holo_size=Width*pit/3;     %%%%% Expect object size

z_d=0.3;  %%%%% Distance from SLM to hologram

%% 02 Object shift
% To avoid occlusion caused by DC term

% Object shift from center
shiftx=0;   shifty=0;   shiftz=0;

%% 03 Shading Effect

% Illumination position / [0 0 0] (if you don't want to put illumination effect)
illu=[1 1 1];

% Continuous Shading
con = 1;    % put or not
    
%% 04 File name - Load File

% DATA STYLE : 1 tri / 1 line ( N by 9 matrix )

obj_name='mesh_teapot.txt';

%% 04 File Name - Save File

% HOLOGRAM FILE NAME (SAVE) : hologram_name
% HOLOGRAM RECONSTRUCTION FILE NAME (SAVE) : reconst_name

hologram_name='plane.jpg';
reconst_name='recon.jpg';

%% %%%%%%%%%%%%%%%%%% Setting done %%%%%%%%%%%%%%%%%% %%

%% Object File Load

obj=load(obj_name);

x=[obj(:,1) obj(:,4) obj(:,7)];
y=[obj(:,2) obj(:,5) obj(:,8)];
z=[obj(:,3) obj(:,6) obj(:,9)]; 

delx=max(max(x))-min(min(x));
dely=max(max(y))-min(min(y));
delz=max(max(z))-min(min(z));

cx = (max(max(x))+min(min(x)))/2; % center in x-axis
cy = (max(max(y))+min(min(y)))/2; % center in y-axis
cz = (max(max(z))+min(min(z)))/2; % center in z-axis

del=[delx,dely,delz];

a=max(del);     % object size

x=(x-repmat(cx,size(obj,1),size(x,2)))./a.*holo_size+shiftx;
y=(y-repmat(cy,size(obj,1),size(x,2)))./a.*holo_size+shifty;
z=(z-repmat(cz,size(obj,1),size(x,2)))./a.*holo_size+shiftz;

obj(:,1)=x(:,1);
obj(:,2)=y(:,1);
obj(:,3)=z(:,1);
obj(:,4)=x(:,2);
obj(:,5)=y(:,2);
obj(:,6)=z(:,2);
obj(:,7)=x(:,3);
obj(:,8)=y(:,3);
obj(:,9)=z(:,3);

%%

dfx=1/pit/Width;
dfy=1/pit/Height;

fx=-(-Width/2:Width/2-1)*dfx;
fy=(-Height/2:Height/2-1)*dfy;
[fx,fy]=meshgrid(fx,fy);
fx=gpuArray(fx);
fy=gpuArray(fy);
fz=sqrt((1/w)^2-fx.^2-fy.^2);

ref=[0 0;1 1;1 0];

U=gpuArray(zeros(Height,Width));
HH=gpuArray(zeros(Height,Width));

%% Find vertex normal vector (just for continuous shading)

if con == 1
[na, nv] = fn_FindVertexNormalVector(obj);
end

%%

for i=1:size(obj,1)

X=reshape(obj(i,:),3,3);

if con == 0
[G, s] = fn_hologram_shading(X, ref, fx, fy, fz, w, illu);
elseif con == 1
G = fn_hologram_continuous(X, ref, fx, fy, fz, w, illu, na, nv, i);
else
    error('"con" must be 1 or 0')
end

if islogical(G)
else

if con == 0
H=G.*exp(1i*2*pi*fz*z_d).*(s+0.1);
elseif con == 1
H=G.*exp(1i*2*pi*fz*z_d);
else
    error('"con" must be 1 or 0')
end

h=fftshift(fft2(fftshift(H)));  % hologram plane
HH=HH+h;

z_d=-z_d;
Ua=fftshift(fft2(fftshift(h)));
Ua=Ua.*exp(-1i.*pi.*w.*z_d.*(fx.^2+fy.^2));
U=U+fftshift(ifft2(fftshift(Ua)));
     % Reconstruction
z_d=-z_d;
end

clc;
display(strcat(num2str(i/size(obj,1)*100),'%'))

end

%%

HH=HH+max(max(abs(HH)));    % HH>0
Inten=abs(HH).^2;
Inten=(Inten-min(min(Inten)))/(max(max(Inten))-min(min(Inten)));    % min=0 , normalize
    % Hologram plane intensity


%% Figure

if fig==0
else
    
axisx=(-Width/2:Width/2-1)*pit;
axisy=(-Height/2:Height/2-1)*pit;

figure(1); imagesc(axisx,axisy,abs(Inten)); colormap(gray); 
figure(2); imagesc(axisx,axisy,abs(U)); colormap(gray);

end

%% Save Files

if sav==0
else
    
plane = (Inten)/max(max((Inten))) * 255;     % gray scale
imwrite(mat2gray(gather(plane)),hologram_name);
        % hologram plane

hologram = abs(U)/max(max(abs(U))) * 255;     % gray scale
imwrite( mat2gray(gather(hologram)) , reconst_name );
        % hologram
end

toc