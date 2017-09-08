clear

load data

%%Generating axis 

      sigma=3.8*10^-8; %sigma=pi/(Ld*z) for Fresnel zone plate
      sigma_f=(3.8*10^-8+4.8*10^-8)/2;
    
      NA_g=0.0025/4.5; %NA for Gaussian low pass filter
      N=1023;
      L=12*10^-3; %lateral length of hologram 
      
	  Ld=633*10^-9;
        %Ld=633*10^-9; %wavelength
      ko=(2*pi)/Ld; %wavenumber
	  dx=L/N; %step size
        
	for m=1:N+1
       
        %Space axis
        x(m)=(m-1)*dx-L/2;
        y(m)=(m-1)*dx-L/2;
        
        %Frequency axis
         kx(m)=(2*pi*(m-1))/(N*dx)-((2*pi*(N))/(N*dx))/2;
         ky(m)=(2*pi*(m-1))/(N*dx)-((2*pi*(N))/(N*dx))/2;      
      
    end
        


%% Convert Horizontal parallax only hologram 

for n=1:N+1
    for m=1:N+1
        
        %Fresnel zone pattern in Frequency Domain        
        FFZP(m,n)=exp(j*sigma*(kx(n).^2+ky(m).^2));
        
        %Fringe matched filte
        F(m,n)=exp(j*sigma_f*(ky(m).^2));

        %Gaussian low pass filter
        G(m,n)=exp((-pi*(Ld/(2*pi*NA_g))^2)*(ky(m).^2));
    
    end
end

F=fftshift(F);
G=fftshift(G);
FFZP=fftshift(FFZP);

%Hologram in Frequency Domain
H=fft2(h);
H=H./max(max(abs(H)));

%Horizontal parallax only hologram in frequency domain
HPO=H.*G.*F;

%Horizontal parallax only hologram in sapce domain
hpo=ifft2(HPO);

%Normalization
hpo=hpo./max(max(abs(hpo)));

