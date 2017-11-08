function [ hologram_wrp ] = fn_wrp( file,lambda,hologram_sampling_interval,hologram_resolution, z_wrp,d)

% % obj                          % object
% % lambda                       % Wave length
% % Hologram_resolution          % Hologram resolution     
% % Hologram_sampling_interval   % Hologram sampling interval
% % z_wrp                        % WRP location


%% setting WRP
load(file);
points_show(obj);
k = 2*pi/lambda;   
z=z_wrp-obj(:,3);

N=round(abs(lambda.*z./(hologram_sampling_interval^2)/2)+0.5).*2-1;        %sampling size of N
Nx = round(obj(:,1)./hologram_sampling_interval)+(hologram_resolution-1)/2;  
Ny = round(obj(:,2)./hologram_sampling_interval)+(hologram_resolution-1)/2;

size_obj = length(obj(:,1));
hologram_wrp = zeros(hologram_resolution);

for o = 1: size_obj
    
    fprintf('%d\n',o);  

    [y_run, x_run]= meshgrid((-(N(o)-1)/2:(N(o)-1)/2)*hologram_sampling_interval,(-(N(o)-1)/2:(N(o)-1)/2)*hologram_sampling_interval);
    r = sign(z(o))*sqrt(z(o)^2 + y_run.^2 + x_run.^2);
    Sub_hologram = exp(1j*rand*2*pi)*exp(1j*k*r)./r;   
    
    temp=zeros(hologram_resolution+N(o), hologram_resolution+N(o));    
    temp(Nx(o):Nx(o)+N(o)-1,Ny(o):Ny(o)+N(o)-1)= Sub_hologram;
    hologram_wrp=hologram_wrp+temp((N(o)+1)/2:hologram_resolution+(N(o)-1)/2,(N(o)+1)/2:hologram_resolution+(N(o)-1)/2);  
 
 
end

   figure,imshow(hologram_wrp)
%% Fresnel Propagation

ROWS= hologram_resolution;                                     
COLS= hologram_resolution;
v=hologram_sampling_interval.*(ones(COLS,1)*(-ROWS/2:ROWS/2-1))';
h=hologram_sampling_interval.*(ones(ROWS,1)*(-COLS/2:COLS/2-1));
d=0.05;
WRPHologram = FresnelPropagation(hologram_wrp, hologram_sampling_interval, hologram_sampling_interval, d, lambda);

phaseadd =WRPHologram;
phase_H = angle(phaseadd) + pi;
phase_H_image = uint8(255*phase_H/max(max(phase_H)));
imwrite(phase_H_image, 'wrp_hologram.bmp', 'bmp');

end

