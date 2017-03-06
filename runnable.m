%% Loading and preparing data.
load('mnist_all.mat')

ns = 5000;
[input_data, input_data_labels] = prepare_data(ns);
clearvars -except input_data input_data_labels ns

%% All calculations in one spot
% 2-D and 3-D embedding spaces
nn_normal = 10;
[y2, m2] = lle(input_data, 2, nn_normal);
[y3, m3] = lle(input_data, 3, nn_normal);

% Using different numbers of nearest neighbors
nn_1 = 1;
nn_2 = 5; 
[y2nn1, ~] = lle(input_data, 2, nn_1);
[y3nn1, ~] = lle(input_data, 3, nn_1);
[y2nn2, ~] = lle(input_data, 2, nn_2);
[y3nn2, ~] = lle(input_data, 3, nn_2);

% Using a different nearest neighbor metric
[y22, m22] = lle2(input_data, 2, nn_normal);
[y32, m32] = lle2(input_data, 3, nn_normal);

s = svd(m2);


%%
figure; 
plot_embedding_space_2d(y2, input_data_labels)
title(sprintf('Embedding space, 2D, nn=%d', nn_normal))

figure; hold all;
plot_embedding_space_3d(y3, input_data_labels)
title(sprintf('Embedding space, 3D, nn=%d', nn_normal))

figure;
imagesc(m2)
colorbar
title('Colormap of M')
box on

figure; hold on
plot(s, 'linewidth', 2);
title('Singular values of M')
box on

figure; 
plot_embedding_space_2d(y2nn1, input_data_labels)
title(sprintf('Embedding space, 2D, nn=%d', nn_1))

figure;
plot_embedding_space_3d(y3nn1, input_data_labels)
title(sprintf('Embedding space, 3D, nn=%d', nn_1))

figure; 
plot_embedding_space_2d(y2nn2, input_data_labels)
title(sprintf('Embedding space, 2D, nn=%d', nn_2))

figure;
plot_embedding_space_3d(y3nn2, input_data_labels)
title(sprintf('Embedding space, 3D, nn=%d', nn_2))

figure; 
plot_embedding_space_2d(y22, input_data_labels)
title(sprintf('Embedding space, 2D, nn=%d, different metric', nn_normal))

figure;
plot_embedding_space_3d(y32, input_data_labels)
title(sprintf('Embedding space, 3D, nn=%d, different metric', nn_normal))
%%
num_nn = 10;
% Point inside manifold
p1 = y2(:, randi(length(y2(1,:))));
p1_hd = map_hd(p1, y2, input_data, num_nn);
figure; 
colormap(gray)
imagesc(reshape(p1_hd, 28, 28)')

% Points outside manifold
p1 = 0.5*y2(:, randi(length(y2(1,:)))) + 0.5*y2(:, randi(length(y2(1,:))));
p1_hd = map_hd(p1, y2, input_data, num_nn);
figure; 
colormap(gray)
imagesc(reshape(p1_hd, 28, 28)')

% Linear interpolation in embedding space
p1 = [-0.04321; -0.005089];
p2 = [0.01935; -0.0151];
p_intermed = [linspace(p1(1), p2(1), 25); linspace(p1(2), p2(2), 25)];

figure                                          
colormap(gray)
for i = 1:25                                    
    subplot(5,5,i)
    p_intermed_hd = map_hd(p_intermed(:,i), y2, input_data, num_nn); 
    imagesc(reshape(p_intermed_hd, 28, 28)')                  
end


% Linear interpolation in original space
p1_hd = map_hd(p1, y2, input_data, num_nn);
p2_hd = map_hd(p2, y2, input_data, num_nn);
p_intermed = zeros(length(p1_hd(:,1)), 25);
for i=1:length(p1_hd(:,1))
    p_intermed(i,:) = linspace(p1_hd(i), p2_hd(i), 25);
end

figure                                          
colormap(gray)                                  
for i = 1:25                                    
    subplot(5,5,i)   
    imagesc(reshape(p_intermed(:,i), 28, 28)')                  
end

