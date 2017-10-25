function [complex_output] = fn_ri2c(realimag)
[Ny, Nx] = size(realimag);
realpart = realimag(1:(Ny/2),:);
imagpart = realimag((Ny/2+1):Ny,:);
complex_output = realpart+1j*imagpart;
end