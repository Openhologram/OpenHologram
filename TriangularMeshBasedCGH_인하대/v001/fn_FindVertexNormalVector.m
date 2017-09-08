function [na,nv] = fn_FindVertexNormalVector(obj)

no=zeros(size(obj,1),3);
for i=1:size(obj,1)
    no(i,:)=cross(obj(i,1:3)-obj(i,4:6),obj(i,7:9)-obj(i,4:6));
end
na=no/norm(no);


Objv=reshape(obj',3,3*size(obj,1))';
nv=zeros(size(Objv));
for i=1:size(Objv,1)
    if Objv(i,:)==[0 0 0]
    else
         
    indexs=ismember(Objv,Objv(i,:),'rows');
    index=find(indexs);
   
    sum=0;
    for j=1:size(index)
        sum=sum+na(fix((index(j)-1)/3)+1,:);
        Objv(index(j),:)=[0 0 0];
    end
    average=sum/j;
    average=average/norm(average);
    for j=1:size(index)
        nv(index(j),:)=average;
    end
    end
end
nv=reshape(nv',9,size(obj,1))';

end