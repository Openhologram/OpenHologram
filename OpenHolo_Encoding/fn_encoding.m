function encodedData = fn_encoding(complexField, method, varargin)

p=inputParser;
defaultPassBand = 'left';
validPassBand = {'left','right','top','bottom'};
checkPassBand = @(x) any(validatestring(x,validPassBand));

addRequired(p,'complexField',@isnumeric)
addRequired(p,'method',@ischar)
addParameter(p,'passband',defaultPassBand, @ischar)

parse(p, complexField, method, varargin{:})

if strcmp(p.Results.method,'ssb')
    encodedData = ssb(p.Results.complexField, p.Results.passband);
elseif strcmp(p.Results.method,'numerical interference')
    encodedData = ni(p.Results.complexField);
elseif strcmp(p.Results.method, 'phase to amplitude')
    encodedData = p2a(p.Results.complexField);
elseif strcmp(p.Results.method, 'real')
    encodedData = real(p.Results.complexField);
elseif strcmp(p.Results.method, 'burckhardt')
    encodedData = burckhardt(p.Results.complexField);
elseif strcmp(p.Results.method, 'two phase')
    encodedData = twoPhase(p.Results.complexField);
end
end

function result = ssb(complexField, passBand)
    [Ny, Nx] = size(complexField);    
    AS = fn_myfft2(complexField);
    
    if strcmp(passBand, 'left')
        AS(:,floor(Nx/2):Nx) = 0;
    elseif strcmp(passBand, 'right')
        AS(:,1:floor(Nx/2)) = 0;
    elseif strcmp(passBand, 'top')
        AS(floor(Ny/2):Ny,:) = 0;
    elseif strcmp(passBand, 'bottom')
        AS(1:floor(Ny/2),:) = 0;
    end
 
    filteredComplexField = fn_myifft2(AS);
    realPart = real(filteredComplexField);
    realPositive = realPart - min(realPart(:));
    result = realPositive/max(realPositive(:));
end

function result = ni(complexField)
    [Ny, Nx] = size(complexField);
    refField = max(abs(complexField(:)));
    interference = abs(complexField + refField).^2;
    result = interference/max(interference(:));
end

function result = p2a(complexField)
    result=angle(complexField)+pi/(2*pi);
end

function result = burckhardt(complexField)
    [Ny, Nx] = size(complexField);
    A1=zeros(Ny,Nx); A2=zeros(Ny,Nx); A3=zeros(Ny,Nx);
    complexField = complexField/max(abs(complexField(:)));
    phase = angle(complexField)+pi;
    amplitude = abs(complexField);
    
    index = find(phase>=0 & phase<2*pi/3);
    A1(index) = amplitude(index).*(cos(phase(index)) + sin(phase(index))/sqrt(3));
    A2(index) = 2*sin(phase(index))/sqrt(3);
    
    index = find(phase>=2*pi/3 & phase<4*pi/3);
    A2(index) = amplitude(index).*(cos(phase(index)-2*pi/3) + sin(phase(index)-2*pi/3)/sqrt(3));
    A3(index) = 2*sin(phase(index)-2*pi/3)/sqrt(3);
    
    index = find(phase>=4*pi/3 & phase<2*pi);
    A3(index) = amplitude(index).*(cos(phase(index)-4*pi/3) + sin(phase(index)-4*pi/3)/sqrt(3));
    A1(index) = 2*sin(phase(index)-4*pi/3)/sqrt(3);    
    
    result=zeros(Ny,3*Nx);
    result(:,1:3:end) = A1;
    result(:,2:3:end) = A2;
    result(:,3:3:end) = A3;    
end

function result = twoPhase(complexField)
    [Ny, Nx] = size(complexField);
    complexField = complexField/max(abs(complexField(:)));
    amplitude = abs(complexField);
    phase = angle(complexField) + pi;
    
    delPhase = acos(amplitude);
    A1 = phase + delPhase;
    A2 = phase - delPhase;
    
    result = zeros(Ny, 2*Nx);
    result(:,1:2:end) = A1;
    result(:,2:2:end) = A2;
end