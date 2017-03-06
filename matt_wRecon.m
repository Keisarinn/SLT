function [W,Wchk] = matt_wRecong(data,K,neighb)

% each row is a new training sample set

% 
[N,D] = size(data);
Z = zeros(K,N);

% nrm = zeros(N,N);
% for j = 1:K
%     for jj = 1:N
%         nrm(neighb(j,jj),jj) = 1;
%     end
% end
W = zeros(K,N);
for n = 1:N
    Z = data(neighb(:,n),:);
%     w = Z'\data(n,:)';
%     w = w./sum(w);
%     W(n,neighb(:,n)) = w;
    Z = Z - repmat(data(n,:),K,1);
    C = Z * Z';
    w = C\ones(K,1);
    w(isinf(w)) = 1;
    w(isnan(w)) = 1;
%     W = W.*nrm;
%     w(n,:) = W(n,:);
    w = w./sum(w);
    W(n,neighb(:,n)) = w;
    Wchk(n,:) = w;
end

end