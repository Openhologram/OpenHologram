function diff = FresnelPropogation(k0,v,h,z,I1_a)
FI = fft2(I1_a);
PROP=exp(1i*k0*(v.^2+h.^2)/(2*z));
FH = fft2(PROP);
diff = fftshift(ifft2(FI.*FH));
