function y = TVnorm(x)
y = sum(sum(sqrt(diffh(x).^2+diffv(x).^2)));
