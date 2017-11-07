function G = fn_AS_continuous(X, ref, fx, fy, fz, w, illu, na, nv, i)

n=na(i,:);

if n(3)>0
if n==[0, 0, 0]      % rotationR error
    G=0;
else

%% global -> local rotation

if n(1)==0 && n(3)==0
    th=0;
else
    th=atan(n(1)/n(3));
end

ph=atan(n(2)/sqrt(n(1)^2+n(3)^2));

% x-axis ph rotation -> y-axis -th rotation
R=[cos(th) 0 -sin(th);
    -sin(ph)*sin(th) cos(ph) -cos(th)*sin(ph);
    cos(ph)*sin(th) sin(ph) cos(ph)*cos(th)];     % global -> local rotation

Xl=R*X;
c=-Xl(:,1);   % shift to match to (0,0)
Xl=Xl+repmat(c,1,3);   % global -> local

%% A

if Xl(1,3)*Xl(2,2)==Xl(2,3)*Xl(1,2)     % rotationA error
    G=0;
else

Xld=Xl(1:2,:)';

% ex) Xl(2,1), ref(2,1) ; x coordinate of 2nd vertex

b=Xld(3,1)*Xld(2,2)-Xld(3,2)*Xld(2,1);

A=zeros(2,2);
for l=1:2
    for m=1:2
        A(l,m)=(ref(3,l)*Xld(2,3-m)-ref(2,l)*Xld(3,3-m))/(b*(-1)^(m+1));
    end
end

%%   f -> fl
    
flx=R(1,1).*fx+R(1,2).*fy+R(1,3).*fz;
fly=R(2,1).*fx+R(2,2).*fy+R(2,3).*fz;
flz=sqrt(complex((1/w)^2-flx.^2-fly.^2));


%% Carrier wave ( each mesh )

uxl=R(1,:);
uyl=R(2,:);
uc=[0;0;1];

du=(1/w).*[uxl;uyl]*uc;

flux=flx-du(1);
fluy=fly-du(2);

%% 

tA=inv(A');

flxA=tA(1,1)*flux+tA(1,2)*fluy;
flyA=tA(2,1)*flux+tA(2,2)*fluy;

%% G0 (including illumination effect)
av=zeros(1,3);
for n=1:3
    av(n) = nv(i,3*(n-1)+1)*illu(1)+nv(i,3*(n-1)+2)*illu(2)+nv(i,3*(n-1)+3)*illu(3)+0.1;
    if av(n)<0
        av(n)=0;
    end
end


a1 = av(1);  a2 = av(3);  a3 = av(2);   % each vertex

D1 = exp(-1i*2*pi.*(flxA+flyA)).*(1i-2*pi.*(flxA+flyA))./(8*pi^3.*flyA.*(flxA+flyA).^2) ...
    +exp(-1i*2*pi.*flxA).*(2*pi.*flxA-1i)./(8*pi^3.*flxA.^2.*flyA) ...
    +(1i.*(2.*flxA+flyA))./(8*pi^3.*flxA.^2.*(flxA+flyA).^2);
D2 = exp(-1i*2*pi.*(flxA+flyA)).*(1i.*(flxA+2.*flyA)-2*pi.*flyA.*(flxA+flyA))./(8*pi^3.*flyA.^2.*(flxA+flyA).^2) ...
    +exp(-1i*2*pi.*flxA).*(-1i)./(8*pi^3.*flxA.*flyA.^2) ...
    +1i./(8*pi^3.*flxA.*(flxA+flyA).^2);
D3 = -exp(-1i*2*pi.*(flxA+flyA))./(4*pi^2.*flyA.*(flxA+flyA)) ...
    +exp(-1i*2*pi.*flxA)./(4*pi^2.*flxA.*flyA) ...
    -1./(4*pi^2.*flxA.*(flxA+flyA));

D1(flxA==-flyA) = (-2*pi.*flxA(flxA==-flyA)+1i)./(8*pi^3.*flxA(flxA==-flyA).^3).*exp(-1i*2*pi.*flxA(flxA==-flyA)) ...
    -(1i*2*pi^2.*flxA(flxA==-flyA).^2+1i)./(8*pi^3.*flxA(flxA==-flyA).^3);
D2(flxA==-flyA) = (-1i)./(8*pi^3.*flxA(flxA==-flyA).^3).*exp(-1i*2*pi.*flxA(flxA==-flyA)) ...
    +(-1i*2*pi^2.*flxA(flxA==-flyA).^2+1i+2*pi.*flxA(flxA==-flyA))./(8*pi^3.*flxA(flxA==-flyA).^3);
D3(flxA==-flyA) = (-1i)./(4*pi^2.*flxA(flxA==-flyA).^2).*exp(-1i*2*pi.*flxA(flxA==-flyA)) ...
    +(-1i*2*pi.*flxA(flxA==-flyA)+1)./(4*pi^2.*flxA(flxA==-flyA).^2);

D1(flxA~=0 & flyA==0) = (1i*4*pi^2.*flxA(flxA~=0 & flyA==0).^2+4*pi.*flxA(flxA~=0 & flyA==0)-2*1i)./(8*pi^3.*flxA(flxA~=0 & flyA==0).^3).*exp(-1i*2*pi.*flxA(flxA~=0 & flyA==0)) ...
    +1i./(4*pi^3.*flxA(flxA~=0 & flyA==0).^3);
D2(flxA~=0 & flyA==0) = 1/2*D1(flxA~=0 & flyA==0);
D3(flxA~=0 & flyA==0) = ((1i*2*pi.*flxA(flxA~=0 & flyA==0)+1).*exp(-1i*2*pi.*flxA(flxA~=0 & flyA==0))-1)./(4*pi^2.*flxA(flxA~=0 & flyA==0).^2);
    
D1(flxA==0 & flyA~=0) = -(2*pi.*flyA(flxA==0 & flyA~=0)+1).*exp(-1i*2*pi.*flyA(flxA==0 & flyA~=0))./(8*pi^3.*flyA(flxA==0 & flyA~=0).^3) ...
    +(-1i*2*pi^2.*flyA(flxA==0 & flyA~=0).^2+1)./(8*pi^3.*flyA(flxA==0 & flyA~=0).^3);
D2(flxA==0 & flyA~=0) = (-pi.*flyA(flxA==0 & flyA~=0)+1i)./(4*pi^3.*flyA(flxA==0 & flyA~=0).^3).*exp(-1i*2*pi.*flyA(flxA==0 & flyA~=0)) ...
    -(1i+pi.*flyA(flxA==0 & flyA~=0))./(4*pi^3.*flyA(flxA==0 & flyA~=0).^3);
D3(flxA==0 & flyA~=0) = exp(-1i*2*pi.*flyA(flxA==0 & flyA~=0))./(2*pi.*flyA(flxA==0 & flyA~=0))+(1-1i)./(2*pi.*flyA(flxA==0 & flyA~=0));

D1(flxA==0 & flyA==0) = 1/3;
D2(flxA==0 & flyA==0) = 1/6;
D3(flxA==0 & flyA==0) = 1/2;



G0 = (a2-a1).*D1 + (a3-a2).*D2 + a1.*D3;

%% G0 -> Gl -> G -> H -> U

J=1/det(A);   % ref -> tri
Gl=J.*G0;
    % triangle's AS in local domain

Gl=Gl*exp(-1i*2*pi/w*uc'*(R'*c)); 

J=flz./fz;    % local -> global
G=Gl.*J.*exp(1i*2*pi.*(flx*c(1)+fly*c(2)+flz*c(3)));

end
end
else
    G=0;
end
end
