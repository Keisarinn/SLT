function [input_data, input_data_labels] = prepare_data(ns)
%% [input_data, input_data_labels] = prepare_data(data_path)

load('mnist_all.mat')

tr = [train0(1:ns/10,:); train1(1:ns/10,:); train2(1:ns/10,:); train3(1:ns/10,:); ...
      train4(1:ns/10,:); train5(1:ns/10,:); train6(1:ns/10,:); train7(1:ns/10,:); ...
      train8(1:ns/10,:); train9(1:ns/10,:)]';
tr = double(tr);
tr_labels = [0*ones(ns/10,1); 1*ones(ns/10,1); 2*ones(ns/10,1); 3*ones(ns/10,1); ...
             4*ones(ns/10,1); 5*ones(ns/10,1); 6*ones(ns/10,1); 7*ones(ns/10,1); ...
             8*ones(ns/10,1); 9*ones(ns/10,1)];
rperm = randperm(length(tr(1,:))); % shuffling
input_data = tr(:, rperm);
input_data_labels = tr_labels(rperm);


end

