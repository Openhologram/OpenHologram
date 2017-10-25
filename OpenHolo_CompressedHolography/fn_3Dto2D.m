function y = fn_3Dto2D(x)
[Ny, Nx, Nz] = size(x);
y = zeros(Ny, 0);
for index=1:Nz,
    y = [y, x(:,:,index)];
end
end