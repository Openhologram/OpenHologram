function output=fn_myfft(input)
output = fftshift(fft(ifftshift(input)));