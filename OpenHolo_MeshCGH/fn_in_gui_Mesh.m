function fn_main_Mesh_gui(input,file)

% object load
loadname = fullfile(file.loadPath,file.loadName);
obj = load(loadname);
obj = fn_normalizeCenteringObj(obj);
obj = fn_scaleShiftObj(obj, [str2double(input.objectsize.String), str2double(input.objectsize.String), str2double(input.objectsize.String)], [str2double(input.shiftx.String), str2double(input.shifty.String), str2double(input.shiftz.String)]);

% hologram generation
holo.dx = str2double(input.dx.String);   holo.dy = str2double(input.dy.String);     holo.Nx = str2double(input.Nx.String);   holo.Ny = str2double(input.Ny.String);       
holo.wavelength = str2double(input.wavelength.String);
shading.con = input.continuousbutton.Value;     shading.illu = [str2double(input.lightx.String), str2double(input.lighty.String), str2double(input.lightz.String)];
hologram = fn_genH(obj, holo, shading, input.checkgpu.Value);

% numerical reconstruction
[rec,du,dv] = fn_FresnelPropagation_as(hologram, holo.dx, holo.dy, str2double(input.shiftz.String), holo.wavelength, input.checkgpu.Value);



% figure
if input.figurecheck.Value==0
else
    
axisx=(-holo.Nx/2:holo.Nx/2-1)*holo.dx;
axisy=(-holo.Ny/2:holo.Ny/2-1)*holo.dy;

axes(input.axes1); imagesc(axisx,axisy,abs(hologram)); colormap(gray); colorbar(); title('Hologram amplitude') 
axes(input.axes2); imagesc(axisx, axisy, angle(hologram)); colormap(gray); colorbar(); title('Hologram phase')
axes(input.axes3); imagesc(axisx,axisy,abs(rec)); colormap(gray); colorbar(); title('Reconstruction amplitude') 

end

% save
if input.savecheck.Value==0
else
    
hologramAbsNormalized255 = 255*abs(hologram)/max(abs(hologram(:)));     % gray scale
recAbsNormalized255 = 255*abs(rec)/max(abs(rec(:)));     % gray scale

if input.checkgpu.Value==1
    hologramAbsNormalized255 = gather(hologramAbsNormalized255);
    recAbsNormalized255 = gather(recAbsNormalized255);
end
imwrite(mat2gray(hologramAbsNormalized255), input.holosavefile.String);       % hologram plane
imwrite(mat2gray(recAbsNormalized255) , input.reconsavefile.String);        % hologram
end