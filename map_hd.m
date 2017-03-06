function phd = map_hd(p, y, x, num_nn)
%% phd = map_hd(p, y, x, nn)

dm1 = @(x1,x2) exp(-(sum((x1-x2).^2)));
dm2 = @(x1,x2) sqrt(sum((x1-x2).^2));
[nn_vecs, nn_ix] = nn(p, y, num_nn);
w = zeros(length(y(1,:)),1);
for i=1:length(nn_ix)
    w(nn_ix(i)) = dm1(p, nn_vecs(:,i));
end
w = w./sum(w);
phd = x*w;

end

function [nn_vecs, nn_ix] = nn(p, x, num_nn)
%% [nn_vecs, nn_ix] = lle_nn_1(p, x, num_nn)

    dum = bsxfun(@minus, x, p);
    dum = dum.^2;
    dum = sum(dum,1);
    [~, ix] = sort(dum,'ascend');
    nn_ix = ix(1:num_nn);
    nn_vecs = x(:,nn_ix);

end

