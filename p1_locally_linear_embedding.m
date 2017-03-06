N = 1000;

images = loadMNISTImages('data/t10k-images-idx3-ubyte');
labels = loadMNISTLabels('data/t10k-labels-idx1-ubyte');

[Y, M] = lle(images(:,1:N), 3, 3);

figure(1)
%scatter(Y(1,:), Y(2,:), 30, labels(1:N), 'filled')
scatter3(Y(1,:), Y(2,:), Y(3,:), 30, labels(1:N), 'filled')

caxis([-0.5, 9.5])
b = [   0.0 0.0 1.0;
        0.1 0.4 0.9;
        0.2 0.6 0.8;
        0.3 0.8 0.8;
        0.4 1.0 0.6;
        0.6 1.0 0.4;
        0.7 0.8 0.3;
        0.8 0.6 0.2;
        0.9 0.4 0.1;
        1.0 0.0 0.0];
colormap(b);
colorbar()

figure(2)
imshow(M)

figure(3)
plot(svd(M))