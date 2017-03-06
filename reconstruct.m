function [img A]=reconstruct(y,mappedX,X,k)
% y point to reconstruct
% mappedX projected points 
% original points
[IDX,D] = knnsearch(mappedX,y,'K',k);
X=X';
A=X(IDX,:);
D=D/sum(D);
img=D*A;
