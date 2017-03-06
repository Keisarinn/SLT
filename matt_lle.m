function [Y,M] = matt_lle(data,K,d)
tic
neighb = matt_knn(data,K);
t1 = toc;
disp({'time for neighb = ',t1})
% weights
[W,Wchk]  = matt_wRecon(data,K,neighb);

% embedding coordinates
[N,NN] = size(W);
M = (eye(N) - W)'*(eye(N) - W);
[V,D] = eigs(M,d+1,0); %only keeps d+1 eigenvectors
V = fliplr(V);
Y = V(:,2:end);
Y = Y.*sqrt(N);
Y = Y';

end
