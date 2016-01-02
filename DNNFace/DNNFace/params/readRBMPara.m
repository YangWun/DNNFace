%used for reading RBM model parameter
clear all;
% 
% fid=fopen('norm-persistent-lvl1.rbm','r');
% W=fread(fid,9216*1024,'float');
% W=reshape(W,[9216, 1024]);
% WT=fread(fid,9216*1024,'int');
% WT=reshape(WT,[1024, 9216]);
% A=fread(fid,9216,'float');
% B=fread(fid,1024,'float');
% VW=fread(fid,9216*1024,'float');
% VW=reshape(VW, [9216,1024]);
% Q=fread(fid,1024,'float');

visible_size=48*48;
hidden_size=32*32;
filename='facial-persistent-lvl1.rbm';
fid=fopen(filename,'r');
W=fread(fid,visible_size*hidden_size,'float');
W=reshape(W,[visible_size, hidden_size]);
WT=fread(fid,visible_size*hidden_size,'float');
WT=reshape(WT,[hidden_size, visible_size]);
A=fread(fid,visible_size,'float');
B=fread(fid,hidden_size,'float');
VW=fread(fid,visible_size*hidden_size,'float');
VW=reshape(VW, [visible_size, hidden_size]);
Q=fread(fid,hidden_size,'float');






