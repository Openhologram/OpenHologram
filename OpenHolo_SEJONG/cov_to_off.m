function [ OUT_H ] = cov_to_off(row,col,H_c,H_s,ang_x,ang_y,l_x,l_y,wl)
%*************************************************************
% Convert complex hologram to off-axis hologram
% row,col is pixel # of axis x and y
% H_c,H_s are input sin and cos hologram data
% ang_x, ang_y are angle of off-axis
% l_x,l_y are length of back ground(field of view) of axis x and y
% wl is wavelength
%
% ex) offh=cov_to_off(x,y,H,ang_x,anx_y,l_x,l_y,wl);
%*************************************************************

% Generation complex hologram
H=gen_holo_data(row,col,H_c,H_s);

% Generation axis space domain
r = linspace(1,row,row); c = linspace(1,col,col);
[x,y]=meshgrid(l_x/(row-1)*(r-1)-l_x/2, l_y/(col-1)*(c-1)-l_y/2);

% Convert off-axis hologram
offh = H.*exp(1i*((2*pi)/wl)*(x*sin(ang_x)+y*sin(ang_y)));

% Extracting real part
H=real(offh);

% Put dc term
H = H - min(min(H));

% Normalization
OUT_H=H./max(max(H));

% Save result data
imwrite(OUT_H,'Result_data\Off-axis_H.bmp');

end

