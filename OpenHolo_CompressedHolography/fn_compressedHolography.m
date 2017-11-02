function reconstructions = fn_compressedHolography(complexField, dx, dy, lambda, z, maxIter)

% Add path for TwIST functions
addpath('./TwISTfunctions');    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TwIST functions under the directory ./TwISTfunctions/ were created by
%
% J. Bioucas-Dias and M. Figueiredo, "A New TwIST: Two-Step
% Iterative Shrinkage/Thresholding Algorithms for Image 
% Restoration",  IEEE Transactions on Image processing, 2007.
% 
% and
% 
% J. Bioucas-Dias and M. Figueiredo, "A Monotonic Two-Step 
% Algorithm for Compressive Sensing and Other Ill-Posed 
% Inverse Problems", submitted, 2007.
%
% Authors: Jose Bioucas-Dias and Mario Figueiredo, October, 2007.
% 
% Please check for the latest version of the code and papers at
% www.lx.it.pt/~bioucas/TwIST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% functionals for TwIST
nz = length(z);
A = @(x) fn_volume2plane(fn_2Dto3D(fn_ri2c(x),nz), dx, dy, z, lambda);  
AT = @(x) fn_plane2volume(fn_ri2c(x), dx, dy, -z, lambda);

tv_iters = 5;
Psi = @(x,th)  tvdenoise(x,2/th,tv_iters);
Phi = @(x) TVnorm(x);

%% TwIST 
tau = 0.05; %0.01 
tolA = 1e-6;

g = fn_c2ri(complexField);

[f_reconstruct,dummy,obj_twist,...
    times_twist,dummy,mse_twist]= ...
    TwIST(g,A,tau,...
    'AT', AT, ...
    'Psi', Psi, ...
    'Phi',Phi, ...
    'Initialization',2,...
    'Monotone',1,...
    'StopCriterion',1,...
    'MaxIterA',maxIter,...
    'MinIterA',maxIter,...
    'ToleranceA',tolA,...
    'Verbose', 1);

reconstructions = fn_2Dto3D(fn_ri2c(f_reconstruct),nz);
end