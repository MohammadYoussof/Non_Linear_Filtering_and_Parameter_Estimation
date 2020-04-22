function [ y ] = gauss_dist(x,meu,sigma )
%GAUSS_DIST function for gaussian distribution
    y=(1/(sqrt(2*pi*sigma)))*exp((-(x-meu).^2)/(2*sigma));
end
