function OUT_H = chr_abb_comp(row, col,H_c,H_s, z, R, wl)
%*************************************************************
% Chromatic aberration compensation filter
% row, col are pixel # of axis x and y
% H_c,H_s are input sin and cos hologram data of RGB, [red, green, blue]
% z is focal locations of RGB, [red, green, blue]
% R is radius of curvature of lens
% wl is wavelengths of RGB, [red, green, blue]
%
% ex) comp_H=chr_abb_comp(x,y,H,z,l_x,l_y,wl);
%
%*************************************************************            

    % Generation complex hologram
    IN_H=gen_holo_data(row,col,H_c,H_s);

    for i=1:3
        H = IN_H(:,:,i);            
        % Generation CAC filter
        [~, FFZP] = gen_FZP(row,col,z(1)-z(i),R,R,wl(i));
        FFZP=fftshift(FFZP);
        
        % Complex hologram in frequency domain
        FH=fft2(H);
        
        % Compensated hologram
        FH_CAC=FH.*conj(FFZP);
        
        % CAC hologram in space domain
        IFH_CAC=ifft2(FH_CAC); 
        OUT_H(:,:,i) = IFH_CAC;
    end
end