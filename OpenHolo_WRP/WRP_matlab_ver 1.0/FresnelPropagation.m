function hologram = FresnelPropagation(field, dx, dy, z, lambda)
    k = 2*pi/lambda;
    [Ny, Nx] = size(field);
    
%     temp_x = [1:Nx]-floor((Nx+1)/2);
%     temp_y = [1:Ny]-floor((Ny+1)/2);
    fx = 1./(Nx*dx);
    fy = 1./(Ny*dy);  
    x = ones(Ny,1)*[0:floor((Nx-1)/2) -ceil((Nx+1)/2)+1:-1]*fx;          %Note order of points for FFT
    y = [0:floor((Ny-1)/2) -ceil((Ny+1)/2)+1:-1]'*ones(1,Nx)*fy;
 
    H = exp(1i*k*z).*exp(-1i*pi*lambda*z*(x.^2+y.^2));   %Fourier transform of h
    O = fft2(field);                                   %Fourier transform of o
    hologram =ifft2(O.*H);          
    
end