load('mnist_all.mat');

%% parameters

%% data 
d = 2;
numSamp = 250;
clear data;
clear lbl
data = [];
lbl = [];
for k = 0:9
   tmp = strcat('train',num2str(k));
   datatmp = eval(tmp);
   datatmp = double(datatmp);
   data = cat(1,data,datatmp(1:numSamp,:));
   lbl = cat(1,lbl,k*ones(numSamp,d));
end

K = 14;
[Y,M] = matt_lle(data,K,d);
% 
% K = 20;
% [Y12,~] = matt_lle(data,K,d);
% 
% K = 24;
% [Y16,~] = matt_lle(data,K,d);
% 
% K = 32;
% [Y32,~] = matt_lle(data,K,d);


%%
[Y,M] = matt_lle(data,20,2);
%%

[U,S,V] = svd(M);
%%

s = diag(S);

%% visualize 2D
Y = Y;
mrk = {'om','oc','or','og','ob','+m','+k','oy','+c','ok'};
% {'om','oc','or','og','ob','*m','*c','*r','*g','*b'};
for i = 0:9
    hold on
    plot(Y(1,1+numSamp*i:numSamp*(i+1)),Y(2,1+numSamp*i:numSamp*(i+1)),mrk{i+1});
end

legend('0','1','2','3','4','5','6','7','8','9')

%% visualize 3D

mrk = {'om','oc','or','og','ob','+m','+k','oy','+c','ok'};
for i = 0:9
    hold on
    scatter3(Y(1,1+numSamp*i:numSamp*(i+1)),...
        Y(2,1+numSamp*i:numSamp*(i+1)),...
        Y(3,1+numSamp*i:numSamp*(i+1)),...
        mrk{i+1});
end

legend('0','1','2','3','4','5','6','7','8','9')


%% go back

pRand = [0.44412;0.21234];

d = sum((repmat(pRand,1,2500) - Y).^2);
[val,idx] = sort(d,2);

kHat = 10;

A = Y(:,idx(1:kHat));
WZ = exp(-5.*(1:kHat));
w = pRand\A;
w = WZ.*w;
w = w./sum(w);

xHat = sum(repmat(w',1,784).*data(idx(1:kHat),:));

xHat = reshape(xHat,28,28);


