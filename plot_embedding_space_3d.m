function plot_embedding_space_3d(y, labels)
%% plot_embedding_space_3d(y, labels)
hold all
c = {'xy', 'xr', 'xg', 'xb', 'xk', 'oy', 'or', 'og', 'ob', 'ok'}; 
for i=1:10
    dum = labels == (i-1);
    scatter3(y(1,dum), y(2,dum), y(3,dum), c{i})
end
legend('0','1','2','3','4','5','6','7','8','9')
title('Embedding space, 3D')
box on
grid on

end

