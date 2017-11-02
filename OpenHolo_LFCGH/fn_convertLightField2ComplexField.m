function complexField = fn_convertLightField2ComplexField(lightField)
[Nv,Nu,Nt,Ns] = size(lightField);
complexField = zeros(Nv*Nt, Nu*Ns);
for idxS=1:Ns
    for idxT=1:Nt
        uRange = (1:Nu) + (idxS-1)*Nu;
        vRange = (1:Nv) + (idxT-1)*Nv;
        complexField(vRange,uRange)=fn_myfft2(lightField(:,:,idxT,idxS).*exp(1j*rand(Nv,Nu)*2*pi));
    end
end
end