function [y, m] = lle2(x, d, num_nn)
%% [y, m] = lle2(x, d, num_nn)
% Another version of lle using a different nearest neighbor metric.
%
%   Input:
%       x: Matrix of input data. Vectors are the columns of the matrix.
%       d: The dimensionality of the embedding space.
%       num_nn: The number of nearest neighbours used in the algorithm.
%
%   Output:
%       y: Matrix of output data. Vectors are the columns of the matrix.
%       m: m = (I-w)'*(I-w)

    nn = lle_nn_v2(x, num_nn);
    w = lle_w(x, nn);
    [y, m]= lle_y(w, d);

end

function [nn_vecs, nn_ix] = lle_nn_1_v2(p, x, num_nn)
%% [nn_vecs, nn_ix] = lle_nn_1(p, x, num_nn)

    dum = bsxfun(@minus, x, p);
    dum = abs(dum);
    dum = sum(dum,1);
    [~, ix] = sort(dum,'ascend');
    nn_ix = ix(1:num_nn);
    nn_vecs = x(:,nn_ix);

end

function nn = lle_nn_v2(x, num_nn)
%% nn = lle_nn2(x)

    nn = zeros(length(x(1,:)), num_nn);
    for i=1:length(x(1,:))
        % Adding one to num_nn so it skips itself
        [~, nn_ix] = lle_nn_1_v2(x(:,i), x, num_nn+1);
        nn(i, 1:num_nn) = nn_ix(2:end);    
    end

end

function w = lle_w(x, nn)
%% w = lle_w(x, nn)

    w = zeros(length(x(1,:)), length(x(1,:)));
    for i=1:length(x(1,:))
        nn_vecs = x(:, nn(i,:));
        xi = x(:,i);
        dum = bsxfun(@plus, -nn_vecs, xi);
        c = dum' * dum;
        wi = c/1;
        wi = wi/sum(wi);
        w(i,nn(i,:)) = wi';
    end

end

function [y, m] = lle_y(w, d)
%% y = lle_y(w)

    m = (eye(size(w))-w)' * (eye(size(w))-w);
    [e_vecs, e_vals] = eig(m);
    e_vals_vec = diag(e_vals);
    [~, ix] = sort(e_vals_vec, 'ascend');
    y = e_vecs(:,ix(2:d+1))';

end