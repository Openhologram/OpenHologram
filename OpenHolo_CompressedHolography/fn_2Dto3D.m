function y=fn_2Dto3D(x, Nz)
[Ny, N] = size(x);
Nx = N/Nz;
y = zeros(Ny, Nx, Nz);
for index=1:Nz,
    y(:,:,index) = x(:, (1:Nx)+Nx*(index-1));
end
end