function output=fn_myifft(input)
output = fftshift(ifft(ifftshift(input)));