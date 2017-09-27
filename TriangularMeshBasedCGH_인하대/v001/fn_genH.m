function complexField = fn_genH(obj, holoParam, shadingParam, gpuAvailable)

dfx=1/holoParam.dx/holoParam.Nx;
dfy=1/holoParam.dy/holoParam.Ny;

fx=-(-holoParam.Nx/2:holoParam.Nx/2-1)*dfx;
fy=(-holoParam.Ny/2:holoParam.Ny/2-1)*dfy;
[fx,fy]=meshgrid(fx,fy);
GG=zeros(holoParam.Ny,holoParam.Nx);

if gpuAvailable==1
    fx=gpuArray(fx);
    fy=gpuArray(fy);
    GG=gpuArray(GG);
end

fz=sqrt((1/holoParam.wavelength)^2-fx.^2-fy.^2);
ref=[0 0;1 1;1 0];
    

%% Find vertex normal vector (just for continuous shading)

if shadingParam.con == 1
[na, nv] = fn_FindVertexNormalVector(obj);
end

%%

for i=1:size(obj,1)
    
    X=reshape(obj(i,:),3,3);
    
    if shadingParam.con == 0
        G = fn_AS_shading(X, ref, fx, fy, fz, holoParam.wavelength, shadingParam.illu);
    elseif shadingParam.con == 1
        G = fn_AS_continuous(X, ref, fx, fy, fz, holoParam.wavelength, shadingParam.illu, na, nv, i);
    else
        error('"shadingParam.con" must be 1 or 0');
    end
    
    if islogical(G)
    else
        GG=GG + G;
    end
    
    clc;
    display(strcat(num2str(i/size(obj,1)*100),'%'))
    
end

%%
complexField = fftshift(fft2(fftshift(GG)));  % hologram plane
