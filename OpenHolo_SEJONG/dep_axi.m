function [H,Ab_yn_half] = dep_axi(row,col,H_c,H_s,l_x,l_y,wl)
%*************************************************************
% Extraction of distance parameter using axis transfomation
% row, col are pixel # of axis x and y
% H_c,H_s are input sin and cos hologram data
% l_x,l_y is field of view of axis x and y
% wl is wavelength
%**************************************************************

% Generation complex hologram
H=gen_holo_data(row,col,H_c,H_s);

r = linspace(1,row,row); c = linspace(1,col,col);
[kx,ky]=meshgrid(2*pi*(r-1)/l_x-pi*(row-1)/l_x,2*pi*(c-1)/l_y-pi*(col-1)/l_y);

% Create window function(Gaussian filter)
NA_g=0.025;
G=exp((-pi*(wl/(2*pi*NA_g))^2)*(kx.^2+ky.^2));

%Real-only hologram generation stage
    %Extracting real and imaginary parts of the complex hologram 
    FIr=real(fft2(real(H)));
    FIi=real(fft2(imag(H)));
    %Synthesize the real-only hologram 
    Hsyn=FIr+1i*FIi;
    
    %Gaussian low-pass filtering
    Hsyn=Hsyn.*G;
    
    %Power fringe-adjusted filtering
    Fo=(Hsyn.*Hsyn)./( abs(Hsyn).*abs(Hsyn)+10^-300);
    Fo=fftshift(Fo);
    
%Axis transformation
    t=linspace(0,1,row/2+1);
    tn=t.^0.5;
    Fon=real(Fo(row/2,row/2:row));
    yn=interp1(t,Fon,tn);

%location extraction
Ab_yn=abs(fft(yn));
Ab_yn_half=Ab_yn(row/4:row/2);
[~,z] = max(Ab_yn_half);
Z=-((z-120)/10)/140+0.1;
% figure(),plot(Ab_yn_half);
H = rec_holo(H,Z,l_x,l_y,wl);
% figure(2),imagesc(abs(H)), title(['depth = ',num2str(Z)]),colormap gray, axis image;

end