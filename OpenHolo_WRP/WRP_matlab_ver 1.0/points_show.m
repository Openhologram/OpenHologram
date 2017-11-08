function [ output_args ] = points_show( A )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
   
% sipha=-90;
% 
% XRS=[1 0 0;
%      0 cosd(sipha) -sind(sipha);
%      0 sind(sipha) cosd(sipha);];
%  
% YRS=[cosd(sipha) 0 -sind(sipha) ;
%     0 1 0 ;
%     sind(sipha) 0 cosd(sipha) ;];
% 
% ZRS=[cosd(sipha) -sind(sipha) 0 ;
%     sind(sipha) cosd(sipha) 0 ;
%     0 0 1 ;];
% A=A*ZRS;   

X = A(:,1);
Y = A(:,2);
Z = A(:,3);

figure; plot3(X,Y,Z,'.');xlabel ('x');ylabel ('y');zlabel ('z')

xlabel('x')
ylabel('y')
zlabel('z')
end

