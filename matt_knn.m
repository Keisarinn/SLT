function neighb = matt_knn(data,K)

% each row is a new training sample set
if (1)
    
[N,D] = size(data);
data2 = sum(data.^2,2);
dist = repmat(data2,1,N) - 2*data*data' + repmat(data2',N,1); %quadratic

[~,idx] = sort(dist);
neighb = idx(2:(1+K),:);

else
% 
[N,D] = size(data);
dist = zeros(N,N);
for n = 1:N
    dist(n,:) = sum((repmat(data(n,:),N,1) - data).^2,2)';
end

[~,idx] = sort(dist);
neighb = idx(2:K+1,:);

end
end