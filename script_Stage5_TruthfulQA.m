clear; clc; close all;
cd 'D:\Courses\Postdoc\2026\SIGMA\Matlab Codes';

%% ---- config ----
EMB_FILE    = 'emb_raw_mpnet_TruthfulQA.mat';  % ANSWER embeddings (per-model). NOT the question file.
GRADES_FILE = 'TruthfulQA_grades.xlsx';         % .jsonl or .xlsx accepted (use the file that exists)
RESP_FILE   = 'TruthfulQA_responses.xlsx';      % responses workbook (full_answers sheet) for word counts
MAX_WORDS   = 300;                            % drop a question if ANY model answer exceeds this
DROP_QID    = -1;                              % FIX B: -1 disables. (Was 421, which wrongly dropped a
                                               %        valid complete question; qid 420 >300w is handled
                                               %        by the word filter instead.)

%% ---- load ANSWER embeddings (per-model) ----
S = load(EMB_FILE);
assert(isfield(S, 'Y'), ...
    ['%s has no field Y. Stage 7 needs per-model ANSWER embeddings ' ...
     '(emb_raw_mpnet.mat -> Y), not question embeddings.'], EMB_FILE);
Yraw   = S.Y;                  % cell{M}, each [Nall x dim]
models = S.meta.models;
qids   = S.meta.qids(:);
M      = numel(Yraw);
D      = size(Yraw{1}, 2);

%% ---- grades (loaded once; also used for present_mask) ----
A_tbl = load_grades_nwrong(GRADES_FILE, DROP_QID);   % cols: qid, GroupCount, sum_wrong

% FIX A: present_mask. A question is usable only if ALL M models produced (and
% thus graded) an answer. GroupCount = number of graded models for that qid.
incomplete_qids = A_tbl.qid(A_tbl.GroupCount < M);
if ~isempty(incomplete_qids)
    fprintf('present_mask: dropping %d incomplete question(s) (< %d models): %s\n', ...
        numel(incomplete_qids), M, mat2str(incomplete_qids(:)'));
else
    fprintf('present_mask: all questions answered by all %d models.\n', M);
end

%% ---- word-limit filter: questions where ANY model answer exceeds MAX_WORDS ----
over_qids = over_word_limit_qids(RESP_FILE, MAX_WORDS);
if ~isempty(over_qids)
    fprintf('Word-limit filter (>%d words): dropping %d question(s): %s\n', ...
        MAX_WORDS, numel(over_qids), mat2str(over_qids(:)'));
else
    fprintf('Word-limit filter (>%d words): no questions exceed limit.\n', MAX_WORDS);
end

keep      = (qids ~= DROP_QID) & ~ismember(qids, over_qids) & ~ismember(qids, incomplete_qids);
qids_kept = qids(keep);
nQ        = numel(qids_kept);
fprintf('queries: %d -> %d after drops (Q%d + word-limit + present_mask) | models: %d | dim: %d\n', ...
        numel(qids), nQ, DROP_QID, M, D);

Z = zeros(nQ, D, M);
for i = 1:M
    Mi = Yraw{i}(keep, :);
    if any(any(isnan(Mi)))
        error('NaN in raw answer embeddings (model %s)', models{i});
    end
    Z(:,:,i) = Mi;
end

%% ---- dictionary learning hyperparameters ----
K_max      = M;
spa        = 0.10;
nIter      = 30;
nInits     = 5;
elbow_thr  = 0.05;

%% ---- per-question outputs ----
K_eff   = nan(nQ, 1);
s_max   = nan(nQ, 1);
purity  = nan(nQ, 1);
K_used  = nan(nQ, 1);
recon   = nan(nQ, 1);
R_byK   = nan(nQ, K_max);

CE_sum  = zeros(M, M);
CE_n    = 0;

fprintf('\nStage 7 dictionary learning, spa=%.2f, Q=%d:\n', spa, nQ);
t0 = tic;

for q = 1:nQ
    %% ---- build per-question dim x M matrix, normalize, center ----
    X_q = squeeze(Z(q,:,:));                  % dim x M
    nrm = sqrt(sum(X_q.^2,1)); nrm(nrm==0) = 1;
    Xn  = X_q ./ nrm;                         % each model's answer -> unit norm
    mu  = mean(Xn, 2);                        % consortium mean (the "agreement" component)
    Y_q = Xn - mu;                            % residual = disagreement; rank <= M-1

    %% ---- fit K = 1..K_max with multi-init, pick by elbow ----
    Rk        = nan(K_max, 1);
    Xk_store  = cell(K_max, 1);
    Dk_store  = cell(K_max, 1);
    bestK     = 1;

    for K = 1:K_max
        bestE = inf; bestDk = []; bestXk = [];
        for init = 1:nInits
            if K <= M
                idx = randperm(M, K);
                Di  = Y_q(:, idx);
            else
                Di = [Y_q, randn(D, K-M)];
                Di = Di(:, randperm(K));
            end
            ndi = sqrt(sum(Di.^2,1)); ndi(ndi==0) = 1;
            Di  = Di ./ ndi;

            try
                [Dk, Xki, ~] = my_ACSD(Y_q, Di, spa, nIter);
                e = norm(Y_q - Dk*Xki, 'fro') / max(norm(Y_q,'fro'), eps);
                if e < bestE
                    bestE = e; bestDk = Dk; bestXk = Xki;
                end
            catch
                continue
            end
        end
        Rk(K)        = bestE;
        Dk_store{K}  = bestDk;
        Xk_store{K}  = bestXk;

        if K >= 2 && ~isnan(Rk(K-1)) && (Rk(K-1) - Rk(K)) < elbow_thr
            bestK = K - 1;
            break
        end
        bestK = K;
    end

    R_byK(q, :) = Rk(:)';
    K_used(q)   = bestK;
    recon(q)    = Rk(bestK);
    bestX       = Xk_store{bestK};

    if isempty(bestX)
        K_eff(q)  = 1;
        s_max(q)  = 1;
        purity(q) = 1;
        continue
    end

    %% ---- proper probability-based structural measures ----
    A   = abs(bestX);                    % bestK x M magnitudes
    col = sum(A, 1);                     % 1 x M, total mass per model
    col(col == 0) = 1;
    P   = A ./ col;                      % bestK x M, columns sum to 1
                                         %   P(k, m) = fraction of model m's loading on atom k
    occ = sum(P, 2) / M;                 % bestK x 1, sums to 1
                                         %   occ(k) = average per-model membership on atom k
    occ_pos = occ(occ > 1e-12);

    if isempty(occ_pos)
        K_eff(q)  = 1;
        s_max(q)  = 1;
        purity(q) = 1;
    else
        H         = -sum(occ_pos .* log(occ_pos));
        K_eff(q)  = exp(H);
        s_max(q)  = max(occ);
        purity(q) = mean(max(P, [], 1));
    end

    %% ---- co-endorsement: M x M model-similarity from membership profiles ----
    CE_sum = CE_sum + P' * P;
    CE_n   = CE_n + 1;

    if mod(q, 50) == 0
        fprintf('  q=%4d / %d   K_used=%d   K_eff=%.2f   s_max=%.2f   recon=%.3f   elapsed=%.1fs\n', ...
            q, nQ, bestK, K_eff(q), s_max(q), recon(q), toc(t0));
    end
end
fprintf('Done. Total time: %.1f s\n', toc(t0));

%% ---- per-state aggregation (reuse A_tbl loaded above) ----
fprintf('\ngrades: %d unique qids (DROP_QID=%d)\n', height(A_tbl), DROP_QID);
fprintf('  responses-per-question distribution:\n');
disp(tabulate(A_tbl.GroupCount));

[tf, ig] = ismember(qids_kept, A_tbl.qid);
if any(~tf)
    missing = qids_kept(~tf);
    error('%s missing qids: %s', GRADES_FILE, mat2str(missing(1:min(5,end))'));
end
n_wrong = A_tbl.sum_wrong(ig);

fprintf('  n_wrong distribution:\n');
for k = 0:M
    fprintf('    n_wrong=%d: %d\n', k, sum(n_wrong == k));
end

%% ---- states (auto-scaled by M; reduces to the 6-model scheme) ----
state = strings(nQ, 1);
half  = M / 2;
state(n_wrong == 0)                        = "all_right";
state(n_wrong > 0 & n_wrong <= half - 1)   = "mostly_right";
state(abs(n_wrong - half) < 1)             = "split";
state(n_wrong >= half + 1 & n_wrong < M)   = "mostly_wrong";
state(n_wrong == M)                        = "all_wrong";

states_list = ["all_right","mostly_right","split","mostly_wrong","all_wrong"];
fprintf('\n%-14s %5s %9s %9s %9s %9s %9s %9s\n', ...
    'state','n','K_eff_m','K_eff_sd','s_max_m','purity_m','K_used_m','recon_m');
for s = states_list
    idx = (state == s);
    n   = sum(idx);
    if n > 0
        fprintf('%-14s %5d %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n', ...
            s, n, mean(K_eff(idx)), std(K_eff(idx)), ...
            mean(s_max(idx)), mean(purity(idx)), ...
            mean(K_used(idx)), mean(recon(idx)));
    end
end

%% ---- significance tests: K_eff, s_max, purity (aw vs mw, aw vs ar) + delta ----
% K_eff alone does not discriminate the two regimes (it rises at all_wrong in
% every dataset because it is computed on the mean-subtracted residual). The
% discriminating sparse signal is s_max (shared dominant-mode share) and purity
% (per-model commitment): on a shared-misconception dataset (TruthfulQA) both
% rise at all_wrong (models consolidate onto a SHARED mode); on dispersed-failure
% datasets (SimpleQA/FreshQA) s_max is flat or falls (each model commits to its
% OWN distinct mode). Compare the printed delta(aw-mw) sign for s_max ACROSS the
% three datasets -- that is the cross-dataset sign check.
sig_measures = {'K_eff', K_eff; 's_max', s_max; 'purity', purity};
fprintf('\nMann-Whitney + delta (all_wrong vs mostly_wrong / all_right):\n');
fprintf('  %-8s %12s %12s %12s\n', 'measure', 'p(aw vs mw)', 'p(aw vs ar)', 'd(aw-mw)');
sig = struct();
for sidx = 1:size(sig_measures,1)
    nm = sig_measures{sidx,1};
    v  = sig_measures{sidx,2};
    a  = v(state=="all_wrong");
    w  = v(state=="mostly_wrong");
    rr = v(state=="all_right");
    pmw = ranksum(a, w);
    par = ranksum(a, rr);
    dmw = mean(a,'omitnan') - mean(w,'omitnan');
    fprintf('  %-8s %12.4g %12.4g %+12.4f\n', nm, pmw, par, dmw);
    sig.(nm) = struct('p_aw_mw', pmw, 'p_aw_ar', par, 'delta_aw_mw', dmw, ...
                      'mean_aw', mean(a,'omitnan'), 'mean_mw', mean(w,'omitnan'), ...
                      'mean_ar', mean(rr,'omitnan'));
end
% backward-compatible scalars (K_eff) retained for the save block
p_aw_mw = sig.K_eff.p_aw_mw;
p_aw_ar = sig.K_eff.p_aw_ar;

%% ---- soft co-endorsement matrix ----
CE = CE_sum / max(CE_n, 1);
fprintf('\nSoft co-endorsement matrix (mean P''*P across %d questions):\n', CE_n);
for i = 1:M, fprintf('  %d: %s\n', i, models{i}); end
disp(CE);

%% ---- hierarchical clustering of models by cosine of membership profile ----
fprintf('\nHierarchical clustering of models by (1 - cos(membership profile)):\n');
try
    rnorm = sqrt(sum(CE.^2, 2));
    rnorm(rnorm == 0) = 1;
    Cn = CE ./ rnorm;
    Csim = Cn * Cn';
    Csim = max(min(Csim, 1), -1);
    Cdist = 1 - Csim;
    Cdist = Cdist - diag(diag(Cdist));   % force zero diagonal for squareform
    Cdist = max(Cdist, 0);
    dvec = squareform(Cdist, 'tovector');
    Zlink = linkage(dvec, 'average');
    order = optimalleaforder(Zlink, dvec);
    for k = 1:M
        fprintf('  pos %d: %s\n', k, models{order(k)});
    end
catch ME
    fprintf('  (clustering skipped: %s)\n', ME.message);
end

%% ---- save ----
if ~exist('out','dir'), mkdir('out'); end
save('out/stage7_dictionary_TruthfulQA.mat', ...
     'K_eff','s_max','purity','K_used','recon','R_byK', ...
     'CE','state','qids_kept','models','spa','nIter','nInits','K_max','n_wrong', ...
     'over_qids','incomplete_qids','MAX_WORDS','p_aw_mw','p_aw_ar','sig', ...
     '-v7.3');
fprintf('\nSaved out/stage7_dictionary.mat\n');

%% ---- diagnostic figure ----
figure('Position',[100 100 1200 400]);
subplot(1,3,1);
boxplot(K_eff, state, 'GroupOrder', cellstr(states_list));
ylabel('K_{eff}(q)'); ylim([0.8 K_max+0.5]); grid on;
title(sprintf('K_{eff} per state (spa=%.2f)', spa));

subplot(1,3,2);
boxplot(s_max, state, 'GroupOrder', cellstr(states_list));
ylabel('s_{max}(q)'); ylim([0 1.05]); grid on;
title('Dominant-atom share per state');

subplot(1,3,3);
boxplot(K_used, state, 'GroupOrder', cellstr(states_list));
ylabel('K_{used}(q)'); ylim([0.5 K_max+0.5]); grid on;
title('Selected K per state');

saveas(gcf, 'out/stage7_Keff_smax_by_state.png');
fprintf('Saved figure: out/stage7_Keff_smax_by_state.png\n');


%% ============================================================
%  Local functions
%% ============================================================
function over_qids = over_word_limit_qids(resp_file, max_words)
    % Returns the question_ids where AT LEAST ONE model answer exceeds
    % max_words. Reads the 'full_answers' sheet (one answer per model).
    if exist(resp_file, 'file') ~= 2
        error(['Responses file not found: "%s"\n' ...
               'Looked in: %s\n' ...
               'Set RESP_FILE to the actual file, or use a full absolute path.'], ...
               resp_file, pwd);
    end
    R = readtable(resp_file, 'Sheet', 'full_answers', 'VariableNamingRule', 'preserve');
    vn       = R.Properties.VariableNames;
    ans_cols = vn(~ismember(vn, {'question_id','question'}));   % model answer columns
    nR = height(R);
    mw = zeros(nR, 1);
    for r = 1:nR
        w = 0;
        for c = 1:numel(ans_cols)
            val = R.(ans_cols{c})(r);                           % cell or string scalar
            w   = max(w, numel(regexp(string(val), '\S+', 'match')));
        end
        mw(r) = w;
    end
    over_qids = R.question_id(mw > max_words);
    over_qids = over_qids(:);
end

function A_tbl = load_grades_nwrong(fname, drop_qid)
    % Returns a table with columns: qid, GroupCount, sum_wrong
    % wrong := (verdict == FALSE). Accepts .jsonl or .xlsx.
    if exist(fname, 'file') ~= 2
        error(['Grades file not found: "%s"\n' ...
               'Looked in: %s\n' ...
               'Set GRADES_FILE to the actual file (run  dir(''*grades*'')  to list), ' ...
               'or use a full absolute path.'], fname, pwd);
    end
    [~,~,ext] = fileparts(fname);
    if strcmpi(ext, '.jsonl')
        fid = fopen(fname, 'r');
        if fid < 0, error('Cannot open %s (in %s)', fname, pwd); end
        qid = []; wrong = [];
        tline = fgetl(fid);
        while ischar(tline)
            s = strtrim(tline);
            if ~isempty(s)
                r = jsondecode(s);
                qid(end+1,1)   = r.question_id;                                %#ok<AGROW>
                wrong(end+1,1) = double(strcmpi(string(r.verdict),'FALSE'));    %#ok<AGROW>
            end
            tline = fgetl(fid);
        end
        fclose(fid);
        T = table(qid, wrong, 'VariableNames', {'qid','wrong'});
    else
        G = readtable(fname);
        v = G.verdict;
        if iscell(v) || isstring(v) || ischar(v)
            wrong = double(strcmpi(string(v), 'FALSE'));
        else
            wrong = double(~logical(v));     % numeric/logical verdict: TRUE=correct
        end
        T = table(G.question_id, wrong, 'VariableNames', {'qid','wrong'});
    end
    A_tbl = groupsummary(T, 'qid', 'sum', 'wrong');
    A_tbl = A_tbl(A_tbl.qid ~= drop_qid, :);
end
