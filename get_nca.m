clear
clc
format compact

RAND_SEED = 1;
rng(RAND_SEED,'twister');

addpath(genpath('functions'));
addpath('nca');

save_path = './results/';

dataset = 'reuters_r8'; 
MAX_DICT_SIZE = 50000; 

% Optimization parameters
max_iter = 1; % number of iterations
save_frequency = max_iter; % frequency of saving results
batch = 32;   % batch size in batch gradient descent (B in the paper)
range = 200;  % neighborhood size (N in the paper)
lr_w = 1e+1; % learning rate for w
lr_A = 1e+0;  % learing rate for A
lambda = 1000; % parameter in regularized transport problem (lambda in the paper)
projected_dim = 30;
cv_folds = 1; % number of folds for cross-validation

for split = 1:cv_folds
    save_couter = 0;
    Err_v = [];
    Err_t = [];
    w_all = [];
    A_all = [];
    [xtr,ytr, BOW_xtr, indices_tr, sequence_tr, word_vector] = load_data_python(dataset, 'train', split, 500);
    [xte,yte, BOW_xte, indices_te, sequence_te, ~] = load_data_python(dataset, 'test', split, 200);
    
    ntr = length(ytr);
    nte= length(yte);
    dim = size(xtr{1},1);

    % Compute document center
    xtr_center = zeros(dim, ntr);
    ytr = ytr-1;
    for i  = 1:ntr
        xtr_center(:,i) = xtr{i} * BOW_xtr{i}' / sum(BOW_xtr{i});
    end
    xte_center = zeros(dim, nte);
    for i  = 1:nte
        xte_center(:,i) = xte{i} * BOW_xte{i}' / sum(BOW_xte{i});
    end

    % Load intialize A (trained with WCD)
    %load(['metric_init/', 'bbcsport', '_seed', num2str(split), '.mat'])
    %A = get_nca_matrix(xtr_center,ytr,projected_dim);
    save('center_data.mat','xtr_center','ytr');

end
