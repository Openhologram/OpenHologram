function complexField = fn_getComplexFieldFrom4PhaseShiftedHolograms(I0,I90,I180,I270)

dummySmall = 1e-12;
phase = atan((I270-I90)./(I0-I180+dummySmall));
amplitude = (1/4)*((I0-I180)./cos(phase));
complexField = amplitude.*exp(-1j*phase);

end
