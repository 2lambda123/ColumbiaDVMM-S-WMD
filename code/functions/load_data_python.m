function [xtr, ytr, BOW_xtr, indices_tr, sequence_tr, word_vector] = load_data_python(dataset,subset,seed,number)
    rng(seed);
	train_data = load(['dataset/', dataset,'_',subset,'_python.mat']);
    train_data = train_data.save_object;
    BOW = train_data{1};
    index_list = train_data{2};
    sequence_list = train_data{3};
    label = train_data{4};
    word_vector = train_data{5};
    n_tr = size(label,2);
    for i = 1:numel(index_list)
        index_list{i} = index_list{i}+1;
        sequence_list{i} = sequence_list{i}+1;
    end
    select_list = randperm(n_tr);
    select_list = select_list(1:number);
    xtr = cell(1,number);
    %sequence_tr = cell(1,number);
    for i = 1:number
        xtr{i} = double(word_vector(index_list{select_list(i)},:))';
    end
    ytr = double(label(select_list)+1);
    BOW_xtr = BOW(select_list);
    for i = 1:numel(BOW_xtr)
        BOW_xtr{i} = double(BOW_xtr{i});
    end
    indices_tr = index_list(select_list);
    sequence_tr = sequence_list(select_list);
end
