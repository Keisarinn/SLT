function plot_embedding_space_2d(y, labels)
%% plot_embedding_space_2d(y, labels)
hold all
c = {'xy', 'xr', 'xg', 'xb', 'xk', 'oy', 'or', 'og', 'ob', 'ok'}; 
for i=1:10
    dum = labels == (i-1);
    plot(y(1,dum), y(2,dum), c{i})
end
legend('0','1','2','3','4','5','6','7','8','9')
title('Embedding space, 3D')
box on
grid on

end

