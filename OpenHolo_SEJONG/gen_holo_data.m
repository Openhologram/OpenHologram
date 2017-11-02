function OUT_H  = gen_holo_data(row,col,path_cos_h,path_sin_h)
%*************************************************************
% Generation hologram
% path_sin_h is file name of sin hologram
% path_cos_h is file name of cos hologram
%
% ex) H=gen_holo_data(x,y,'real.bmp','imaginary.bmp','bmp',nf_min(=[x,y]), nf_max(=[x,y]), shift(=[x,y]));
%
%*************************************************************

% load data
cos_H = imread(path_cos_h,'bmp');
sin_H = imread(path_sin_h,'bmp');
[~,~,z]=size(cos_H);
OUT_H=zeros(row,col,z);
for i=1:z 
    cos_h=cos_H(:,:,i);
    sin_h=sin_H(:,:,i);
    cos_h = double(cos_h);
    sin_h = double(sin_h);    

    % normalization
    sin_h=sin_h/mean(mean(sin_h));
    cos_h=cos_h/mean(mean(cos_h));
    % sin_h = normc(sin_h);
    sin_h=sin_h/max(max(abs(sin_h)));
    % cos_h = normc(cos_h);
    cos_h=cos_h/max(max(abs(cos_h)));
    sin_h=sin_h-min(min(sin_h));
    cos_h=cos_h-min(min(cos_h));
    % 
    % Generation hologram
    h = cos_h+1i*sin_h;
    [sx,sy] = size(h);
    H = reshape(h,sqrt(sx*sy),sqrt(sx*sy));
    OUT_H(:,:,i) = extend_holo(row,col,H);
end

end