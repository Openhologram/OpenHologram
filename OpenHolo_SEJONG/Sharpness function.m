load data;

for depth = 1:1:10
    
    exp = ['I = h_',num2str(depth),'';];
    eval(exp)
    
    I = real(I);
    
    % Brenner function
    
    [Sx,Sy]=size(I);
    Fx=zeros(Sx,Sy);Fy=zeros(Sx,Sy);
    
    for i=1:Sx;
        for j=1:Sy-2;
            if abs(I(i,j+1)-I(i,j))>=Th;
                Fy(i,j)=abs(I(i,j+2)-I(i,j)).^2;
            else Fy(i,j)=0;
            end
        end
    end
    for i=1:Sx-2;
        for j=1:Sy;
            if abs(I(i+1,j)-I(i,j))>=Th;
                Fx(i,j)=abs(I(i+2,j)-I(i,j)).^2;
            else Fx(i,j)=0;
            end
        end
    end
    
    Fx_sum=sum(Fx(:));Fy_sum=sum(Fy(:));
    F_3=Fx_sum+Fy_sum;
    F=F_3;    
end
