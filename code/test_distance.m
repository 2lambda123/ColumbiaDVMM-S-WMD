%clear
%clc
%format compact
%addpath(genpath('functions'))

dataset = 'bbcsport'; 
% Optimization parameters

% [xtr,xte,ytr,yte, BOW_xtr,BOW_xte, indices_tr, indices_te] = load_data(dataset, 1);
% 
% load('./distance_data/bbcsport_SWMD_matrix.mat');
% xtr_test = xtr(1);
% ytr_test = ytr(1);
% BOW_xtr_test = BOW_xtr(1);
% indices_tr_test = indices_tr(1);

% Compute loss
SWMD_tr = distance_matrix_swmd(xtr_test, ytr_test, BOW_xtr_test, indices_tr_test, w, 1000, A);
