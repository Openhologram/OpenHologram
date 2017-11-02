function [output] = fn_plane2volume(input, dx, dy, z, lambda)
% input: complex, 2D
% output: real+imag, 2D

[Ny, Nx] = size(input);
Nz = length(z);
output_cube = zeros(Ny, Nx, Nz);
for index = 1:Nz
    [rec,du,dv] = fn_FresnelPropagation_as(input, dx, dy, z(index), lambda, false);
    output_cube(:,:,index) = rec;
end
output = fn_3Dto2D(output_cube);
output = fn_c2ri(output);
end