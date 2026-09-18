clear; clc;
cd 'D:\Courses\Postdoc\2026\SIGMA\Matlab Codes';

%% ---- model-subset config ----
N_MODELS = 9;   % <-- set to 6 or 9 (uses the FIRST N_MODELS in the embedding file's order)
MAX_WORDS = 300;                           % drop a question if ANY model answer exceeds this

%% ---- word-limit filter config ----
RESP_FILE = 'TruthfulQA_responses.xlsx';   % responses workbook; MUST match the dataset
                                           % of the grades file used below

%% ---- load raw inputs ----
fprintf('Loading embeddings...\n');
SA = load('emb_raw_TruthfulQA_bge.mat');
SQ = load('emb_question_TruthfulQA_bge.mat');

Yans       = SA.Y;                 % cell{Mall}, each [Nall x dim]
models_all = SA.meta.models;
qids_emb   = SA.meta.qids(:);
Q_full     = SQ.Q;
qids_qf    = SQ.qids(:);
assert(isequal(qids_emb, qids_qf), 'qid mismatch between Y and Q files');

Mall = numel(Yans);
assert(N_MODELS >= 2 && N_MODELS <= Mall, ...
    'N_MODELS=%d invalid; embedding file has %d models.', N_MODELS, Mall);

% --- select the first N_MODELS ---
sel    = 1:N_MODELS;
Yans   = Yans(sel);
models = models_all(sel);
M      = numel(Yans);
dim    = size(Yans{1}, 2);
fprintf('Using %d of %d models: %s\n', M, Mall, strjoin(string(models), ', '));

%% ---- n_wrong from TruthfulQA_grades.xlsx (model-aware, over SELECTED models) ----
G = readtable('TruthfulQA_grades.xlsx');

% FIX 1: robust verdict parsing.
% The grades file stores verdict as the text "TRUE"/"FALSE", but readtable may
% auto-detect that column as LOGICAL. Handle both so n_wrong is never silently 0.
v = G.verdict;
if islogical(v)
    G_is_wrong = ~v;                          % logical: false == wrong
elseif isnumeric(v)
    G_is_wrong = (v == 0);                     % numeric 0/1: 0 == wrong
else
    G_is_wrong = strcmpi(string(v), 'false');  % text: case-insensitive "false"
end
fprintf('Verdict column class: %s | wrong rows: %d of %d\n', ...
        class(v), sum(G_is_wrong), numel(G_is_wrong));

% Map grade rows to embedding-model indices so n_wrong counts ONLY selected models.
% Tries an exact match first, then a loose match (handles 'haiku' vs
% '~anthropic/claude-haiku-latest', short vs vendor-prefixed names, etc.).
gmodel_str = string(G.model);
sel_names  = string(models);
midx = zeros(height(G), 1);
for r = 1:height(G)
    midx(r) = match_model(gmodel_str(r), sel_names);
end
keep_rows = midx > 0;                          % grade rows belonging to a selected model
fprintf('Grade rows matched to selected models: %d of %d\n', sum(keep_rows), height(G));
if sum(keep_rows) == 0
    error(['No grade-file models matched the selected embedding models.\n' ...
           'Grade model examples: %s\nSelected: %s\n' ...
           'Edit match_model() name handling if these differ.'], ...
           strjoin(unique(gmodel_str(1:min(5,end)))', ', '), strjoin(sel_names, ', '));
end

T = table(G.question_id(keep_rows), double(G_is_wrong(keep_rows)), ...
          'VariableNames', {'qid','wrong'});
A = groupsummary(T, 'qid', {'sum','nnz'}, 'wrong');   % sum_wrong + count available
A.Properties.VariableNames{'sum_wrong'}     = 'n_wrong';
% GroupCount = number of selected models with a grade for that question

% FIX 3: removed stale  A = A(A.qid ~= 421, :);  (qid 421 is a valid 9-model
% question; the >300-word qid 420 is dropped by the word filter below instead).

% FIX 2: present_mask. Keep only questions answered/graded by ALL M selected
% models, so every retained question forms a complete M-model cloud and its
% n_wrong state is comparable. For the 9-model run this drops qids 271 and 406.
n_before = height(A);
A = A(A.GroupCount == M, :);
fprintf('present_mask: kept %d of %d questions graded by all %d models (dropped %d incomplete).\n', ...
        height(A), n_before, M, n_before - height(A));

%% ---- word-limit filter: drop questions where ANY model answer exceeds MAX_WORDS ----
fprintf('Applying word-limit filter (MAX_WORDS=%d) from %s ...\n', MAX_WORDS, RESP_FILE);
R = readtable(RESP_FILE, 'Sheet', 'full_answers', 'VariableNamingRule', 'preserve');
rvars    = R.Properties.VariableNames;
ans_cols = rvars(~ismember(rvars, {'question_id','question'}));   % all model answer columns

wc = @(s) numel(regexp(string(s), '\S+', 'match'));   % whitespace word count
max_words = zeros(height(R), 1);
for r = 1:height(R)
    w = 0;
    for c = 1:numel(ans_cols)
        w = max(w, wc(R.(ans_cols{c})(r)));
    end
    max_words(r) = w;
end

over_qids = R.question_id(max_words > MAX_WORDS);
if ~isempty(over_qids)
    fprintf('  Dropping %d question(s) over %d words: %s\n', ...
            numel(over_qids), MAX_WORDS, mat2str(over_qids(:)'));
else
    fprintf('  No questions exceed %d words.\n', MAX_WORDS);
end
A = A(~ismember(A.qid, over_qids), :);

% canonical qid list = graded qids that are also embedded
[tf, ig] = ismember(A.qid, qids_emb);
if any(~tf), error('TruthfulQA_grades.xlsx has qids missing from embeddings: %s', ...
        mat2str(A.qid(find(~tf,5))')); end
qids    = A.qid;
n_wrong = A.n_wrong;
nQ      = numel(qids);
fprintf('Questions: %d  | models: %d  | dim: %d\n', nQ, M, dim);

%% ---- question embeddings aligned to qids ----
Q = Q_full(ig, :);
Q = Q ./ max(vecnorm(Q, 2, 2), eps);

%% ---- per-question loop: cloud measures + marin + phillips ----
log_vol     = nan(nQ,1);  D_geo  = nan(nQ,1);  R_bar  = nan(nQ,1);
log10_kappa = nan(nQ,1);  PR     = nan(nQ,1);  H_spec = nan(nQ,1);
hamzah      = nan(nQ,1);  marin  = nan(nQ,1);  phillips = nan(nQ,1);
r_qa        = nan(nQ, M);

% Phillips feasibility-adapted settings (see geom_volume / notes)
dprime = min(15, M - 1);
Karch  = min([16, dprime + 1, M]);
rng(0);

Xq = zeros(M, dim);
for t = 1:nQ
    row = ig(t);
    for i = 1:M
        Xq(i, :) = Yans{i}(row, :);
    end
    Xn = Xq ./ max(vecnorm(Xq, 2, 2), eps);     % L2-normalize rows

    % --- geometric cloud measures (+ pair_mean = hamzah) ---
    m = cloud_measures(Xn);
    hamzah(t)      = m(1);
    R_bar(t)       = m(2);
    log10_kappa(t) = m(3);
    log_vol(t)     = m(4);
    D_geo(t)       = m(5);
    PR(t)          = m(6);
    H_spec(t)      = m(7);

    % --- marin: cos(question, answer_m) averaged across models ---
    r_qa(t, :) = (Xn * Q(t, :)')';
    marin(t)   = mean(r_qa(t, :));

    % --- phillips: PCA -> archetypal analysis -> log simplex volume ---
    phillips(t) = geom_volume(Xn, dprime, Karch);
end
fprintf('Phillips: d''=%d, K=%d archetypes/question\n', dprime, Karch);

fprintf('\nn_wrong distribution:\n');
for k = 0:M
    c = sum(n_wrong == k);
    if c, fprintf('  n_wrong=%d: %d\n', k, c); end
end

%% ---- States (auto-scaled by M) ----
state = strings(nQ, 1);
half = M / 2;
state(n_wrong == 0)                        = "all_right";
state(n_wrong > 0 & n_wrong <= half - 1)   = "mostly_right";
state(abs(n_wrong - half) < 1)             = "split";
state(n_wrong >= half + 1 & n_wrong < M)   = "mostly_wrong";
state(n_wrong == M)                        = "all_wrong";
states_list = ["all_right","mostly_right","split","mostly_wrong","all_wrong"];

%% ---- Metrics (geometric first, baselines last; phillips third-last) ----
metrics = { ...
    'log_vol',     log_vol;
    'D_geo',       D_geo;
    'R_bar',       R_bar;
    'log10_kappa', log10_kappa;
    'PR',          PR;
    'H_spec',      H_spec;
    'phillips',    phillips;
    'hamzah',      hamzah;
    'marin',       marin};
nMet = size(metrics, 1);

%% ---- Per-state means ----
fprintf('\nPer-state means:\n');
fprintf('%-14s %5s', 'state', 'n');
for i = 1:nMet, fprintf(' %12s', metrics{i,1}); end
fprintf('\n');
for s = states_list
    idx = (state == s); n = sum(idx);
    if n == 0, continue; end
    fprintf('%-14s %5d', s, n);
    for i = 1:nMet
        fprintf(' %12.4f', mean(metrics{i,2}(idx), 'omitnan'));
    end
    fprintf('\n');
end

%% ---- Per-state stds ----
fprintf('\nPer-state stds:\n');
fprintf('%-14s %5s', 'state', 'n');
for i = 1:nMet, fprintf(' %12s', metrics{i,1}); end
fprintf('\n');
for s = states_list
    idx = (state == s); n = sum(idx);
    if n == 0, continue; end
    fprintf('%-14s %5d', s, n);
    for i = 1:nMet
        fprintf(' %12.4f', std(metrics{i,2}(idx), 'omitnan'));
    end
    fprintf('\n');
end

%% ---- Mann-Whitney ----
fprintf('\nMann-Whitney p-values:\n');
idx_aw = (state == "all_wrong");
idx_mw = (state == "mostly_wrong");
idx_ar = (state == "all_right");
fprintf('%-14s %14s %14s\n', 'metric', 'p(aw vs mw)', 'p(aw vs ar)');
for i = 1:nMet
    x  = metrics{i,2};
    p1 = ranksum(x(idx_aw), x(idx_mw));
    p2 = ranksum(x(idx_aw), x(idx_ar));
    fprintf('%-14s %14.4g %14.4g\n', metrics{i,1}, p1, p2);
end

%% ---- Rebound ----
fprintf('\nRebound (all_wrong - mostly_wrong):\n');
for i = 1:nMet
    x = metrics{i,2};
    d = mean(x(idx_aw), 'omitnan') - mean(x(idx_mw), 'omitnan');
    fprintf('  %-14s  delta = %+.4f\n', metrics{i,1}, d);
end

%% ---- Save ----
save('stage4_by_state_TruthfulQA_bge.mat', 'state', 'n_wrong', 'qids', ...
     'log_vol','D_geo','R_bar','log10_kappa','PR','H_spec', ...
     'phillips','hamzah','marin','r_qa','models','N_MODELS', '-v7.3');
fprintf('\nSaved stage4_by_state_TruthfulQA_bge.mat  (N_MODELS=%d)\n', N_MODELS);


%% ============================================================
%  Local functions
%% ============================================================
function idx = match_model(gname, sel_names)
    % Return the index in sel_names that gname corresponds to, or 0 if none.
    % Exact match first; then loose match on the trailing path segment and on
    % substring containment, to bridge short vs vendor-prefixed model names.
    gname = string(gname);
    % exact
    e = find(sel_names == gname, 1);
    if ~isempty(e), idx = e; return; end
    % normalize: take part after last '/', lowercase, strip leading '~'
    norm = @(s) erase(lower(extractAfter(s + "/", textBoundary("start"))), "~");
    gtail = local_tail(gname);
    for k = 1:numel(sel_names)
        stail = local_tail(sel_names(k));
        if gtail == stail || contains(gtail, stail) || contains(stail, gtail)
            idx = k; return;
        end
    end
    idx = 0;
end

function t = local_tail(s)
    s = string(s);
    if contains(s, "/"), s = extractAfter(s, strlength(s) - strlength(extractAfter(s, "/"))); end
    parts = split(s, "/");
    t = erase(lower(parts(end)), "~");
end

function m = cloud_measures(X)
    % X: M x p, rows already L2-normalized unit vectors.
    % Returns [pair_mean, R_bar, log10_kappa, log_vol, D_geo, PR, H_spec].
    [M, p] = size(X);

    % pairwise cosine matrix (full M x M), blank the diagonal, average the rest
    C = X * X';
    C(logical(eye(M))) = NaN;            % remove self-similarity (diagonal)
    pair_mean = mean(C(:), 'omitnan');   % average of all off-diagonal cosines

    % vMF mean resultant length + Banerjee (2005) concentration
    mu    = mean(X, 1);
    R_bar = min(max(norm(mu), eps), 1 - 1e-12);
    kappa = R_bar * (p - R_bar^2) / (1 - R_bar^2);
    log10_kappa = log10(kappa);

    % spherical angular dispersion about the mean direction (distinct from PR/H_spec)
    mu_dir = mu ./ max(norm(mu), eps);
    dots   = min(max(X * mu_dir', -1), 1);
    D_geo  = mean(acos(dots).^2);

    % spectral descriptors of the centred cloud  (STANDARD defs — edit if yours differ)
    Xc  = X - mu;
    s   = svd(Xc);
    s   = s(s > 1e-9);
    if isempty(s)
        log_vol = NaN; PR = NaN; H_spec = NaN;
    else
        lam = s.^2;
        pk  = lam / sum(lam);
        log_vol = sum(log(s));                  % log generalized volume
        PR      = (sum(lam))^2 / sum(lam.^2);   % participation ratio (effective # dims)
        H_spec  = -sum(pk .* log(pk));          % spectral (Shannon) entropy
    end

    m = [pair_mean, R_bar, log10_kappa, log_vol, D_geo, PR, H_spec];
end

function gv = geom_volume(X, dprime, K)
    % Phillips-style: PCA -> archetypal analysis -> log intrinsic simplex volume.
    Xc = X - mean(X, 1);
    [~, ~, V] = svd(Xc, 'econ');
    r = min(dprime, size(V, 2));
    Z = Xc * V(:, 1:r);
    [~, B] = archetypal_analysis(Z, K, 120);
    arche = B * Z;
    gv = log(simplex_volume(arche) + 1e-12);
end

function v = simplex_volume(P)
    K = size(P, 1);
    if K < 2, v = 0; return; end
    E = P(2:end, :) - P(1, :);
    G = E * E';
    d = det(G);
    if d <= 0, v = 0; return; end
    v = sqrt(d) / factorial(K - 1);
end

function [A, B] = archetypal_analysis(X, K, iters)
    n = size(X, 1);
    K = min(K, n);
    pr = randperm(n);
    B = zeros(K, n);
    for k = 1:K, B(k, pr(k)) = 1; end
    A = ones(n, K) / K;
    for it = 1:iters
        Z  = B * X;
        La = norm(Z)^2; if La < eps, La = eps; end
        for s = 1:10
            grad = (A * Z - X) * Z';
            A = simplex_rows(A - grad / La);
        end
        Lb = (norm(A)^2) * (norm(X)^2); if Lb < eps, Lb = eps; end
        for s = 1:10
            Rres  = A * (B * X) - X;
            gradB = A' * Rres * X';
            B = simplex_rows(B - gradB / Lb);
        end
    end
end

function Sm = simplex_rows(Mm)
    [r, c] = size(Mm);
    U   = sort(Mm, 2, 'descend');
    css = cumsum(U, 2);
    j   = 1:c;
    t   = (css - 1) ./ j;
    rho = max(sum(U > t, 2), 1);
    idx = sub2ind([r, c], (1:r)', rho);
    theta = t(idx);
    Sm = max(Mm - theta, 0);
end
