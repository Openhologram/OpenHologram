clear,clc
%% º¹¿ø
load('data')

fzprr=fft2(fzpr);fzprr=fftshift(fzprr);fzprr2=fft2(conj(fzpr));fzprr2=fftshift(fzprr2);
fzpgg=fft2(fzpg);fzpgg=fftshift(fzpgg);
fzpbb=fft2(fzpb);fzpbb=fftshift(fzpbb);


Re_R=fzprr.*fzprr2;                          %% filter
Re_G=fzpgg.*fzprr2;                          %% filter
Re_B=fzpbb.*fzprr2;                          %% filter

re_g=ifft2(Re_G);
re_g=ifftshift(re_g);

re_b=ifft2(Re_B);
re_b=ifftshift(re_b);

Re_G2=Re_G.*conj(Re_G);
Re_B2=Re_B.*conj(Re_B);

rer=ifft2(Re_R);
rer=ifftshift(rer);

rerm=max(max(abs(rer)));
rer=rer./rerm;

figure(1)
plot(abs(rer(513,513-99:513+99)))
title('Red reconstruction')

figure(2)
plot(abs(re_g(513,:)))
title('Green reconstruction')

figure(3)
plot(abs(re_b(513,:)))
title('Blue reconstruction')

reg=ifft2(Re_G2);
reg=ifftshift(reg);

regm=max(max(abs(reg)));
reg=reg./regm;

figure(4)
plot(abs(reg(513,513-99:513+99)))
title('Green compensation')


reb=ifft2(Re_B2);
reb=ifftshift(reb);

rebm=max(max(abs(reb)));
reb=reb./rebm;

figure(5)
plot(abs(reb(513,513-99:513+99)))
title('Blue compensation')

