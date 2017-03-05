
images = loadMNISTImages('train-images.idx3-ubyte');
%labels = loadMNISTLabels('train-labels.idx1-ubyte');

%display_network(images(:,1:100)); % Show the first 100 images
%%
%b)
[mappedX2, mapping2] = lle(images(:,1:1000)', 2,30);
[mappedX3, mapping3] = lle(images(:,1:1000)', 3,30);
 figure() 
 scatter(mappedX2(:,1),mappedX2(:,2))
 figure() 
scatter3(mappedX3(:,1),mappedX3(:,2),mappedX3(:,3))
%%
%c)
M1=full(mapping3.M);
J1=M1;
J1(J1~=0)=1;
I1 = mat2gray(M1);
M2=full(mapping2.M);
I2 = mat2gray(M2);
J2=M2;
J2(J2~=0)=1;
figure() 
imshow(I1)
figure
imshow(I2)
figure
imshow(J1)
figure() 
imshow(J2)
%%
s = svds(mapping2.M,1000);
[n,xout] = hist(s,30); %Use 30 bins for the histogram
bar(xout,n/sum(n)); %relative frequency is n/sum(n)
xlabel('Singular Value')
ylabel('Relative frequency')
%%
%d)
 figure()
for i=4:8:44
[mappedX2, mapping2] = lle(images(:,1:1000)', 2,i);

 subplot(2,3,(i+4)/8)   
 scatter(mappedX2(:,1),mappedX2(:,2))
 title(i)
end

%%
 figure()
for i=4:8:44
[mappedX2, mapping2] = lle1(images(:,1:1000)', 2,i);

 subplot(2,3,(i+4)/8)   
 scatter(mappedX2(:,1),mappedX2(:,2))
 title(i)
end
%%
 figure()
for i=4:8:44
[mappedX2, mapping2] = lle2(images(:,1:1000)', 2,i);

 subplot(2,3,(i+4)/8)   
 scatter(mappedX2(:,1),mappedX2(:,2))
 title(i)
end

%%
for i=4:8:44
[mappedX2, mapping2] = lle3(images(:,1:1000)', 2,i);

 subplot(2,3,(i+4)/8)   
 scatter(mappedX2(:,1),mappedX2(:,2))
 title(i)
end


% figure
% for i=4:8:44
%  [mappedX3, mapping3] = lle(images(:,1:1000)', 3,i);
%  subplot(2,3,(i+4)/8)   
%  scatter3(mappedX3(:,1),mappedX3(:,2),mappedX3(:,3))
% end
%%
%e)
[mappedX3, mapping3] = lle(images(:,1:1000)', 2,20)
y=[0.1 0];
[img A]=reconstruct(y,mappedX3,images(:,1:1000),17);
figure
display_network([img; A]')
%%
[mappedX3, mapping3] = lle(images(:,1:1000)', 3,20)
y=[0 0 0];
[img A]=reconstruct(y,mappedX3,images(:,1:1000),17);
figure
display_network([img; A]')

%%
%Two neigbors interpolation
[img A]=reconstruct(y,mappedX3,images(:,1:1000),2);
D=[0:0.05:1;1:-0.05:0]';
line_img=D*A;
figure
display_network(line_img')