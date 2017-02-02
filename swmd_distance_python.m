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
    for i  = 1:ntr
        xtr_center(:,i) = xtr{i} * BOW_xtr{i}' / sum(BOW_xtr{i});
    end
    xte_center = zeros(dim, nte);
    for i  = 1:nte
        xte_center(:,i) = xte{i} * BOW_xte{i}' / sum(BOW_xte{i});
    end

    % Load intialize A (trained with WCD)
    %load(['metric_init/', 'bbcsport', '_seed', num2str(split), '.mat'])
    A = get_nca_matrix(xtr_center,ytr,projected_dim);
    % Define optimization parameters
    w = ones(MAX_DICT_SIZE,1);  % weights over all words in the dictionary

    % Test learned metric for WCD
    Dc = distance(xtr_center, xte_center);
    err_wcd = knn_fall_back(Dc,ytr,yte,1:19);
    Dc = distance(A * xtr_center, A * xte_center);
    err_swcd = knn_fall_back(Dc,ytr,yte,1:19);

    tStart = tic;

    % Main loop
    for iter = 1 : max_iter

        fprintf('Dataset: %s  Split: %d  Iteration: %d \n',dataset,split,iter)

        tIterStart = tic;        
        [dw, dA] = grad_swmd(xtr,ytr,BOW_xtr,indices_tr,xtr_center,w,A,lambda,batch,range);

        % Update w and A
        w = w - lr_w * dw;
        lower_bound = 0.01;
        upper_bound = 10;
        w(w<lower_bound) = lower_bound;
        w(w>upper_bound) = upper_bound;
        A = A - lr_A * dA;

        % Compute loss
        filename = [save_path, dataset,'_SWMD_matrix.mat'];
        disp('begin test');
        %disp(iter);
        %disp(save_frequency);
        if mod(iter, save_frequency) == 0
            save_couter = save_couter + 1;
            SWMD_tr = distance_matrix_swmd(xtr, ytr, BOW_xtr, indices_tr, w, lambda, A);
            %SWMD_all = distance_matrix_swmd([xtr, xte], [ytr, yte], [BOW_xtr, BOW_xte], [indices_tr, indices_te], w, lambda, A);
            disp('test over');
            %save(filename,'SWMD_tr', 'SWMD_all', 'xtr','xte','ytr','yte', 'BOW_xtr','BOW_xte',...
            %        'indices_te','indices_tr','w','lambda','A');
        end
        tIterEnd = toc(tIterStart);
    end
end
