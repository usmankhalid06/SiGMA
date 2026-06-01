function embeddings = get_sentence_embeddings(texts, model_name)
    persistent model current_model_name helper

    if nargin < 2
        model_name = 'all-MiniLM-L12-v2';
    end

    % Load Python helper (once)
    if isempty(helper)
        % Create helper Python script in current directory
        fid = fopen('_emb_helper.py', 'w');
        fprintf(fid, 'def make_list(cell_data):\n');
        fprintf(fid, '    return [str(x) for x in cell_data]\n');
        fprintf(fid, '\n');
        fprintf(fid, 'def encode_texts(model, texts_list):\n');
        fprintf(fid, '    embs = model.encode(texts_list)\n');
        fprintf(fid, '    return embs.tolist()\n');
        fclose(fid);
        helper = py.importlib.import_module('_emb_helper');
    end

    % Reload model if different from current
    if isempty(model) || ~strcmp(current_model_name, model_name)
        fprintf('Loading sentence transformer model: %s...\n', model_name);
        st = py.importlib.import_module('sentence_transformers');
        model = st.SentenceTransformer(model_name);
        current_model_name = model_name;
        fprintf('Model loaded!\n');
    end

    % Ensure texts is a cell array
    if ~iscell(texts)
        texts = {texts};
    end

    % Batch encode
    batch_size = 256;
    n_texts = length(texts);
    embeddings = [];
    emb_dim = 0;

    for batch_start = 1:batch_size:n_texts
        batch_end = min(batch_start + batch_size - 1, n_texts);
        fprintf('  Encoding batch %d-%d of %d\n', batch_start, batch_end, n_texts);

        % Convert batch to Python list via helper
        batch_texts = texts(batch_start:batch_end);
        batch_texts = batch_texts(:)';
        py_texts = helper.make_list(batch_texts);

        % Encode
        emb_list = helper.encode_texts(model, py_texts);

        % Convert to MATLAB
        n_batch = int64(py.len(emb_list));

        if emb_dim == 0
            first_row = emb_list{1};
            emb_dim = int64(py.len(first_row));
            fprintf('  Embedding dimension: %d\n', emb_dim);
        end

        batch_embeddings = zeros(n_batch, emb_dim);
        for i = 1:n_batch
            row = emb_list{i};
            for j = 1:emb_dim
                batch_embeddings(i, j) = double(row{j});
            end
        end

        embeddings = [embeddings; batch_embeddings];
    end

    fprintf('  Generated %d embeddings of dimension %d\n', size(embeddings));
end