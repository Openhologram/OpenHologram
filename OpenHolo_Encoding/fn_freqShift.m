function result = fn_freqShift(complexField, shift)
[Ny,Nx]= size(complexField);
AS = fn_myfft2(complexField);
shiftedAS = circshift(AS,shift);
result = fn_myifft2(shiftedAS);