function [FZP, FFZP] = gen_FZP(row,col,z,l_x,l_y,wl)
%*************************************************************
% Generation Fresnel zone plate
% row,col are size of FZP
% l_x,l_y are length of back ground(field of view) of axis x and y
% z is distance from the point object to hologram plane
% wl is wavelength of light
% FZP is Fresnel zone plate on space domain
% FFZP is Fresnel zone plate on frequency domain
%
% ex) [FZP, FFZP] = gen_FZP(x,y,z,l_x,l_y,wl);
%*************************************************************

% display function is sigma*((x-x0)^2+(y-y0)^2). All scales are arbitrary.
% sigma=k0/2*z=pi/(wavelength*z)
% sigmaf=z/2*k0=(z*wavwlength)/4*pi
sigma = pi/(wl*z);
sigmaf = (z*wl)/(4*pi);
 
% Generating axis and meshgrid
r = linspace(1,row,row); c = linspace(1,col,col);
[x,y]=meshgrid(l_x/(row-1)*(r-1)-l_x/2, l_y/(col-1)*(c-1)-l_y/2);
[kx,ky]=meshgrid(2*pi*(r-1)/l_x-pi*(row-1)/l_x,2*pi*(c-1)/l_y-pi*(col-1)/l_y);
 
% Generating FZP
FZP=exp(1j*sigma*(x.^2+y.^2));        % space domain
FFZP=exp(1j*sigmaf*(kx.^2+ky.^2));    % frequency domain

end
