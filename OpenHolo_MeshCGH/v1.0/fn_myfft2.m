function output=fn_myfft2(input)
output = fftshift(fft2(ifftshift(input)));