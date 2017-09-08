%% Convert off-axis hologram
clear
%Load the hologram data
load data

%off-axis carrier generation

n=0; 
for x=-511:512
    n=n+1;
    m=0;
    for y=-511:512
        m=m+1;
H(m,n) = h(m,n).*exp(j*((2*pi)/4)*x);
    end
end

%extracting real part
H=real(H);

% put dc term
H = H - min(min(H));

%normalization
M=max(max(H));
H=H./M;
