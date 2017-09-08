function [hologram,du,dv, max_phase_step] = WRP_FresnelPropagation(object, dx, dy, z, lambda)

    k = 2*pi/lambda;
    [Ny, Nx] = size(object);

    fx = 1./(Nx*dx);
    fy = 1./(Ny*dy);  
    x = ones(Ny,1)*[0:floor((Nx-1)/2) -ceil((Nx+1)/2)+1:-1]*fx;          %Note order of points for FFT
    y = [0:floor((Ny-1)/2) -ceil((Ny+1)/2)+1:-1]'*ones(1,Nx)*fy;
 
    H = exp(1i*k*z).*exp(-1i*pi*lambda*z*(x.^2+y.^2));   %Fourier transform of h
    O = fft2(object);                                   %Fourier transform of o
    hologram =ifft2(O.*H);          
    max_phase_step = max(pi*lambda*z*(x(1, 1:(Nx-1)).^2 - x(1, 2:(Nx)).^2));
    du = dx;
    dv = dy;
    
end