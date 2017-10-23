function outputObj = fn_scaleShiftObj(inputNormalizedCenteredObj, scale, shift)
% inputNormalizedCenteredObj
% scale = [scaleX, scaleY, scaleZ]
% shift = [shiftX, shiftY, shiftZ]

outputObj = zeros(size(inputNormalizedCenteredObj));
outputObj(:,1:3:9) = inputNormalizedCenteredObj(:,1:3:9)*scale(1) + shift(1);
outputObj(:,2:3:9) = inputNormalizedCenteredObj(:,2:3:9)*scale(2) + shift(2);
outputObj(:,3:3:9) = inputNormalizedCenteredObj(:,3:3:9)*scale(3) + shift(3);