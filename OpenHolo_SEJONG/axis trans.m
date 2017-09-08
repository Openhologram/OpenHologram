clear

load data
    
%Gaussian filter generation unit    
radius=900; %Radius of Gaussian fiter in arbitrary unit
L=1*10^-2; %Length of frequency axis in arbitrary unit

for r=1:1024,
   for c=1:1024,
        %compute Gaussian fiter
        
        Kx(r)=L/(255*4)*(r-1)-L/2;
        Ky(c)=L/(255*4)*(c-1)-L/2;
   
        G(r,c)=exp(-radius*4*(Kx(r).^2+Ky(c).^2));
   end
end

%Real-only hologram generation stage
    %Extracting real and imaginary parts of the complex hologram 
    FIr=real(fft2(real(h)));
    FIi=real(fft2(imag(h)));
    
    %Synthesize the real-only hologram 
    Hsyn=FIr+j*FIi;
    
%Gaussian low-pass filtering
    Hsyn=Hsyn.*G;
    
%Power fringe-adjusted filtering
    Fo=(Hsyn.*Hsyn)./( abs(Hsyn).*abs(Hsyn)+10^-300);
    Fo=fftshift(Fo);

%Axis transformation
    t=linspace(0,1,129);
    tn=t.^0.5;
    Fon=real(Fo(512,512:512+128));
    yn=interp1(t,Fon,tn);
    
%location extraction
Ab_yn=abs(fft(yn));

[max max_loc] = max(Ab_yn(65:129))
