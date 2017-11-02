function [ OUT_H ] = cov_to_hor(row,col,H_c,H_s,z,l_x,l_y,wl,red_rat)
%*************************************************************
% Convert complex hologram to horizontal parallax only hologram
% row,col is pixel # of axis x and y
% H_c,H_s are input sin and cos hologram data
% wl is wavelength
% l_x,l_y are length of back ground(field of view) of axis x and y
% z is distance of object
% red_rat is data reduction rate
%
%*************************************************************

% Generation complex hologram
H=gen_holo_data(row,col,H_c,H_s);

% NA_lp=NA*NA_g/sqrt(NA^2+NA_g^2);
% red_rat=NA_lp/NA
NA=l_x/(2*z);
NA_g=NA*red_rat;

% Generating axis and meshgrid
r = linspace(1,row,row); c = linspace(1,col,col);
[~,ky]=meshgrid(2*pi*(r-1)/l_x-pi*(row-1)/l_x,2*pi*(c-1)/l_y-pi*(col-1)/l_y);

% display function is sigma*((x-x0)^2+(y-y0)^2). All scales are arbitrary.
sigmaf = (z*wl)/(4*pi);

% Fringe matched filter
F=exp(1i*sigmaf*(ky.^2));
F=fftshift(F);

% Gaussian low pass filter
G=exp((-pi*(wl/(2*pi*NA_g))^2)*ky.^2);
G=fftshift(G);

% Hologram in Frequency Domain
H=fft2(H);

% Horizontal parallax only hologram in frequency domain
HPO=H.*G.*F;

% Horizontal parallax only hologram in sapce domain
H=ifft2(HPO);

% Normalization
OUT_H=H./max(max(H));

% Save result data
imwrite(real(OUT_H),'Result_data\HPO_H_re.bmp');
imwrite(imag(OUT_H),'Result_data\HPO_H_im.bmp');

end

