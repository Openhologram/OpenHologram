function EX_H = extend_holo(row,col,IN_H)
%*************************************************************
% Extend resolution of hologram
% IN_H is input hologram
% out_x,out_y are output hologram resolution of axis x and y
%
% date - 20170930
%*************************************************************
[x,y] = size(IN_H);

[x1,y2]=meshgrid(1:y,1:x);
[xi,yi]=meshgrid(linspace(1,y,col),linspace(1,x,row));

H=interp2(x1,y2,IN_H,xi,yi);
EX_H=zeros(row,col);
if(H(1,1,1)~=0)
    EX_H=H(1:col,1:row);
else    
    EX_H(1:col-1,1:row-1)=H(2:col,2:row);
end
end