%% ============================================================
%  STAGE 3 (BGE variant) — EMBED EVERYTHING IN ONE SHOT  (9 models)
%
%  Encoder robustness check: identical to the mpnet Stage 3, but embeds with
%  BAAI/bge-base-en-v1.5 (768-d) instead of all-mpnet-base-v2. Outputs are
%  tagged with _bge_ so they never overwrite the mpnet .mat files.
%
%  Reads TruthfulQA_responses.xlsx and embeds:
%    (a) the 9 model raw answers per question  -> emb_raw_bge_TruthfulQA.mat
%    (b) the question texts                    -> emb_question_bge_TruthfulQA.mat
%
%  NOTE ON BGE: BGE models were trained with a retrieval query-instruction
%  prefix. This study measures SYMMETRIC similarity (answer-answer cloud
%  geometry, answer-question alignment), NOT retrieval, so NO prefix is added
%  -- prepending identical instruction text to all nine answers would inflate
%  inter-answer similarity and contaminate the cloud. Embeddings are
%  L2-normalized to the unit sphere, exactly as in the mpnet pipeline, so all
%  downstream geometry code (Stage 4) runs unchanged. Both encoders are 768-d,
%  so nothing downstream needs a dimension change.
%
%  Output files:
%
%  emb_raw_bge_TruthfulQA.mat
%    Y            : 9x1 cell, each [nQ x 768], L2-normalized per vector
%    present_mask : 9 x nQ logical   (false where a model had no answer)
%    meta         : struct with source, models, qids, normalization
%
%  emb_question_bge_TruthfulQA.mat
%    Q            : [nQ x 768], L2-normalized per vector
%    qids         : [nQ x 1]
%    meta         : struct
%
%  Model order: {'deepseek','gemini','haiku','qwen','gpt-mini','granite','llama','grok','stepfun'}
%
%  NOTE: questions missing a model answer (e.g. stepfun on 271/406) get a
%  ZERO vector and present_mask=false for that model. Drop those questions in
%  Stage 4 using present_mask (a zero row is not a unit vector and would
%  contaminate the cloud measures).
%% ============================================================
clear functions
clear; clc; close all;

%% ---- Python bridge ----
try
    pyversion('C:\Users\mukhalid\anaconda3\envs\matlab_env\python.exe');
catch ME
    if contains(ME.message, 'Python is loaded')
        warning('Python already loaded.');
    else
        rethrow(ME);
    end
end

%% ---- Config ----
work_dir   = 'D:\Courses\Postdoc\2026\SIGMA\Matlab Codes';
model_name = 'BAAI/bge-base-en-v1.5';     % 768-d BGE encoder (robustness check)
tag        = 'bge';                        % output-file tag
cd(work_dir);

canonical_models = {'deepseek','gemini','haiku','qwen','gpt-mini','granite','llama','grok','stepfun'};
% map the FULL model slug -> canonical short name
% (robust: avoids model_short slash quirks for x-ai/ and stepfun/)
short2canon = containers.Map( ...
    {'deepseek/deepseek-v4-pro','google/gemini-3.5-flash','~anthropic/claude-haiku-latest', ...
     'qwen/qwen3.7-max','~openai/gpt-mini-latest','ibm-granite/granite-4.1-8b', ...
     'meta-llama/llama-4-scout','x-ai/grok-4.3','stepfun/step-3.7-flash'}, ...
    {'deepseek','gemini','haiku','qwen','gpt-mini','granite','llama','grok','stepfun'});
M = numel(canonical_models);

%% ---- Read long sheet (answers + question text live here) ----
fprintf('Reading TruthfulQA_responses.xlsx (sheet ''long'')...\n');
T = readtable('TruthfulQA_responses.xlsx', 'Sheet','long', ...
              'TextType','string', 'VariableNamingRule','preserve');
fprintf('  %d rows, columns: %s\n', height(T), strjoin(T.Properties.VariableNames, ', '));

qid_all   = double(T.('question_id'));
q_all     = string(T.('question'));
ans_all   = string(T.('answer'));
short_all = string(T.('model'));   % full slug -> mapped via short2canon

%% ---- Canonical qid list (sorted ascending) ----
qids = unique(qid_all);
qids = sort(qids);
nQ   = numel(qids);
fprintf('  unique qids: %d (range %d..%d)\n', nQ, min(qids), max(qids));

qid2row = containers.Map(num2cell(qids), num2cell(1:nQ));

%% ============================================================
%% (a) ANSWERS -> emb_raw_bge_TruthfulQA.mat
%% ============================================================

% Build per-model answer arrays in qid order
answer_text = cell(M, 1);
present_mask = false(M, nQ);
for m = 1:M, answer_text{m} = strings(nQ, 1); end

skipped = 0;
for i = 1:height(T)
    sh = char(short_all(i));
    if ~isKey(short2canon, sh), skipped = skipped + 1; continue; end
    canon = short2canon(sh);
    m = find(strcmp(canonical_models, canon), 1);
    if isempty(m), skipped = skipped + 1; continue; end

    q = qid_all(i);
    if ~isKey(qid2row, q), continue; end
    r = qid2row(q);

    ans_txt = ans_all(i);
    if ismissing(ans_txt) || strlength(ans_txt) == 0, continue; end  % skip empty/failed rows

    answer_text{m}(r) = ans_txt;   % last non-empty wins (handles retry duplicates)
    present_mask(m, r) = true;
end
fprintf('  skipped %d rows with unknown model slug\n', skipped);
fprintf('  present per model:');
for m = 1:M
    fprintf(' %s=%d', canonical_models{m}, sum(present_mask(m,:)));
end
fprintf('\n');

% Embed each model's answers
Y = cell(M, 1);
for m = 1:M
    fprintf('\n=== Embedding %s answers (%d) with %s ===\n', ...
            canonical_models{m}, sum(present_mask(m,:)), model_name);

    txt_in = answer_text{m};
    empty_idx = (strlength(txt_in) == 0);
    txt_in(empty_idx) = " ";

    % NO BGE instruction prefix: symmetric similarity, raw answer text only.
    emb = get_sentence_embeddings(cellstr(txt_in), model_name);
    if size(emb,2) == nQ, emb = emb'; end
    d = size(emb,2);

    nrm = vecnorm(emb, 2, 2);
    nrm(nrm == 0) = 1;
    emb = emb ./ nrm;
    emb(empty_idx, :) = 0;          % missing answers -> zero vector (present_mask=false)

    Y{m} = emb;
    fprintf('  -> Y{%d} is [%d x %d]\n', m, size(Y{m},1), size(Y{m},2));
end
D = size(Y{1}, 2);

meta_raw = struct( ...
    'source',         'TruthfulQA_responses/long', ...
    'text',           'raw_answer', ...
    'model_embedder', model_name, ...
    'dim',            D, ...
    'models',         {canonical_models}, ...
    'qids',           qids, ...
    'normalization',  'L2-per-vector');
meta = meta_raw;
out_raw = sprintf('emb_raw_TruthfulQA_%s.mat', tag);
save(out_raw, 'Y', 'present_mask', 'meta', '-v7.3');
fprintf('\nSaved %s (Y %dx1 cell of [%d x %d])\n', out_raw, M, nQ, D);

%% ============================================================
%% (b) QUESTIONS -> emb_question_bge_TruthfulQA.mat
%% ============================================================

% First-seen question text per qid
q_text = strings(nQ, 1);
seen   = false(nQ, 1);
for i = 1:height(T)
    q = qid_all(i);
    if ~isKey(qid2row, q), continue; end
    r = qid2row(q);
    if seen(r), continue; end
    q_text(r) = q_all(i);
    seen(r)   = true;
end

fprintf('\n=== Embedding %d question texts with %s ===\n', nQ, model_name);
% NO BGE instruction prefix here either (symmetric setting).
emb = get_sentence_embeddings(cellstr(q_text), model_name);
if size(emb,2) == nQ, emb = emb'; end
d = size(emb,2);

% L2-normalize each question vector (consistent with the answers)
nrm = vecnorm(emb, 2, 2);
nrm(nrm == 0) = 1;
Q   = emb ./ nrm;

meta_q = struct( ...
    'source',         'TruthfulQA_responses/long column ''question'' (dedup by qid)', ...
    'text',           'question_text', ...
    'model_embedder', model_name, ...
    'dim',            d, ...
    'qids',           qids, ...
    'normalization',  'L2-per-vector');    % FIXED: was mislabeled 'zscore-per-column'
meta = meta_q;
out_q = sprintf('emb_question_TruthfulQA_%s.mat', tag);
save(out_q, 'Q', 'qids', 'meta', '-v7.3');
fprintf('Saved %s (Q is [%d x %d])\n', out_q, nQ, d);

fprintf('\n=== STAGE 3 (BGE) COMPLETE ===\n');
fprintf('  %s   (%d models x %d questions x %d dims, L2)\n', out_raw, M, nQ, D);
fprintf('  %s  (%d questions x %d dims, L2)\n', out_q, nQ, d);