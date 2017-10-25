function [output] = fn_volume2plane(input_cube, dx, dy, z, lambda)
% input_cube: complex, 3D
% output: real+imag, 2D

[Ny, Nx, Nz] = size(input_cube);
output = zeros(Ny, Nx);
for index = 1:Nz
    [slicefield,du,dv] = fn_FresnelPropagation_as(input_cube(:,:,index), dx, dy, z(index), lambda, false);
    output = output + slicefield;
end
output = fn_c2ri(output);
end