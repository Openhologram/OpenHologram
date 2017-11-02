function OUT_H = rec_holo(IN_H,z,l_x,l_y,wl)
%*************************************************************
% Reconstruction hologram
% IN_H : complex hologram
% z is position from hologram plane to reconstruction hologram.
% l_x,l_y are length of back ground(field of view) of axis x and y
% wl is wavelength
%
% ex) RH=rec_holo(H,z,l_x,l_y,wl);
%
%*************************************************************
[x,y,color] = size(IN_H);
OUT_H=zeros(x,y,color);
for i=1:color 
% Generation FZP
[~, FFZP] = gen_FZP(x,y,z,l_x,l_y,wl(i));
FFZP=fftshift(FFZP);

% Input hologram in frequency domain
FH=fft2(IN_H(:,:,i));

% Reconstruction hologram
FHI=FH.*conj(FFZP);

% Reconstructed hologram in space donaim
H=ifft2(FHI);
% OUT_H(:,:,i)=H;   % Non-nomalization
OUT_H(:,:,i)=H./max(max(H));   % nomalization
end

end