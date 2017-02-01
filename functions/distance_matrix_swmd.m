function [WMD] = distance_matrix_swmd(xtr, ytr, BOW_xtr, indices_tr, w, lambda, A)
ntr = length(ytr);

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 1e-4;
end

if ~exist('A', 'var') || isempty(A)
    A = eye(size(xtr{1},1));
end

WMD = zeros(ntr,ntr);
parfor i = 1:ntr
    disp([num2str(i) ' done']);
    Wi = zeros(1,ntr);
    xi    = xtr{i};
    bow_i = BOW_xtr{i}';
    a = bow_i .*w(indices_tr{i});
    a = a / sum(a);
    for j = 1:ntr
        xj    = xtr{j};
        bow_j = BOW_xtr{j}';
        b =bow_j.*w(indices_tr{j});
        b = b / sum(b);
        D  = distance(A*xi, A*xj);
        D(D < 0) = 0;
        D = full(D); 
        [alpha, beta, T, dprimal, ddual] = sinkhorn(D, a, b, lambda, 200, 1e-3);
        Wi(j) = dprimal;
    end
    WMD(i,:) = Wi;
end
