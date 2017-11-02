function tiled2D = fn_showcube(inputcube, nfh)
[Ny, Nx, Nz] = size(inputcube);
nfv = ceil(Nz/nfh);
maxvalue = max(abs(inputcube(:)));

border = 5;
Nyb = Ny+border*2;
Nxb = Nx+border*2;
inputwborder = maxvalue*ones(Nyb, Nxb, Nz);
inputwborder( (1:Ny)+border, (1:Nx)+border, :) = inputcube;

tiled2D = zeros(Nyb*nfv, Nxb*nfh);

for i=1:Nz
    indexy = ceil(i/nfh);
    indexx = i - (indexy-1)*nfh; 
    tiled2D( (1:Nyb)+Nyb*(indexy-1), (1:Nxb)+Nxb*(indexx-1) ) = inputwborder(:,:,i);
end
