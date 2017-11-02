function outputObj = fn_normalizeCenteringObj(inputObj)

x=[inputObj(:,1) inputObj(:,4) inputObj(:,7)];
y=[inputObj(:,2) inputObj(:,5) inputObj(:,8)];
z=[inputObj(:,3) inputObj(:,6) inputObj(:,9)]; 

delx=max(max(x))-min(min(x));
dely=max(max(y))-min(min(y));
delz=max(max(z))-min(min(z));

cx = (max(max(x))+min(min(x)))/2; % center in x-axis
cy = (max(max(y))+min(min(y)))/2; % center in y-axis
cz = (max(max(z))+min(min(z)))/2; % center in z-axis

del=[delx,dely,delz];

a=max(del);     % object size

x=(x-repmat(cx,size(inputObj,1),size(x,2)))./a;
y=(y-repmat(cy,size(inputObj,1),size(x,2)))./a;
z=(z-repmat(cz,size(inputObj,1),size(x,2)))./a;

outputObj = zeros(size(inputObj));
outputObj(:,1:3:9)=x;
outputObj(:,2:3:9)=y;
outputObj(:,3:3:9)=z;
