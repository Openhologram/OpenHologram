function [OUT_H,z,F] = dep_shar(row,col,H_c,H_s, l_x, l_y, wl, z_max, z_min, samp, th)
%*************************************************************
% Extraction of distance parameter using sharpness functions
% row,col is pixel # of axis x and y
% H_c,H_s are input sin and cos hologram data
% l_x,l_y are field of view of axis x and y
% wl is wavelength
% z_max, z_min are maximum and minimum value of distance on z axis
% samp is count of search step
% th is threshold value
%
% ex) Z=dep_shar(H,l_x,l_y,wl,z_max,z_min,samp,th);
%
%**************************************************************

% Generation complex hologram
H=gen_holo_data(row,col,H_c,H_s);
% depth period
dz = (z_max-z_min)/samp;

F = linspace(1,samp,samp);
z = linspace(1,samp,samp);

for n = 1:1:samp+1
    z(n)=(n-1)*dz+z_min;
    % Reconstruction complex hologram
    HI = rec_holo(H,z(n),l_x,l_y,wl);
%     figure(n),imagesc(abs(HI)), title(['depth = ',num2str(z(n))]),colormap gray, colorbar,axis image;
    I = real(HI);
    
%     Brenner function
    [Sx,Sy]=size(I);
    F_I=zeros(Sx,Sy);
    
    for i=1:Sx-2;
        for j=1:Sy-2;            
            if abs(I(i+2,j)-I(i,j))>=th;
                F_I(i,j)=abs(I(i+2,j)-I(i,j)).^2;
            else                
                if abs(I(i,j+2)-I(i,j))>=th;
                    F_I(i,j)=abs(I(i,j+2)-I(i,j)).^2;                 
                end                
            end
        end
    end
    
    F(n)=-sum(sum(F_I));
    
%     F(n)=entropy(I);
end
% figure(99),plot(z,F);

[~, index] = max(F);
Z = z(index);
OUT_H = rec_holo(H,Z,l_x,l_y,wl);
% figure(999),imagesc(abs(OUT_H)), title(['depth = ',num2str(Z)]),colormap gray, axis image;

imwrite(real(OUT_H),'sf_H_c.bmp');
imwrite(real(OUT_H),'sf_H_s.bmp');

end 