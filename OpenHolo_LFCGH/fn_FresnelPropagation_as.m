function [output,du,dv] = fn_FresnelPropagation_as(input, dx, dy, z, lambda, gpuAvailable)
    k = 2*pi/lambda;
    [Ny Nx] = size(input);

    Nxx = 2*Nx;
    Nyy = 2*Ny;
    
    input2x = zeros(Nyy, Nxx);
    if gpuAvailable == 1
        input2x = gpuArray(input2x);
    end

    start_x = round(Nx/2) - 1;
    start_y = round(Ny/2) - 1;
    input2x(start_y + (1:Ny), start_x + (1:Nx)) = input;
    
    dal = 1./(Nxx*dx);  % delta alpha over lambda
    dbl = 1./(Nyy*dy);  % delta beta over lambda

    [al, bl] = meshgrid(((1:Nxx)-Nxx/2)*dal, ((1:Nyy)-Nyy/2)*dbl);

    A = fftshift(fft2(ifftshift(input2x)));
    
    prop_kernel = exp(1j*2*pi*z*sqrt(1/lambda^2 - al.^2 - bl.^2));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % K. Matsushima et al., "Band-limited angular spectrum method for
    % numerical simulation of free-space propagation in far and near
    % fields," Opt. Express 17, 19662 (2009) 참조
    % FresnelPropagationShift_as.m 과 동일. 다만 sx=sy=0
    sx = 0;
    sy = 0;
    fla = abs(-al*z./sqrt(1/lambda^2 - al.^2 - bl.^2) + sx);
    flb = abs(-bl*z./sqrt(1/lambda^2 - al.^2 - bl.^2) + sy);
    index = fla > 1/(2*dal);
    prop_kernel(index)=0;
    index = flb > 1/(2*dbl);
    prop_kernel(index)=0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

    intermediate = fftshift(ifft2(ifftshift(A.*prop_kernel)));
    
    output = intermediate(start_y + (1:Ny), start_x + (1:Nx));
    
    du = dx;
    dv = dy;
end