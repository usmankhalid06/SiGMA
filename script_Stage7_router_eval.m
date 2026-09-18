%% ============================================================
%  STAGE 7 — Router vs consensus-gate evaluation  (fixed pooled masks)
%
%  Consensus-magnitude gate vs a query-type router (myth-prone -> always verify).
%  On AW clouds the consensus answer is WRONG, so "accept" = dangerous false-accept.
%
%  Inputs:
%     stage4_by_state_<DS>.mat   (state, n_wrong, qids, hamzah = rho, ...)
%     <DS>_mythprone.xlsx        (question_id, myth_prone)  from Stage 6
%
%  Thresholds calibrated on AR only; classifier sees only question text -> non-circular.
%% ============================================================
clear; clc;
cd 'D:\Courses\Postdoc\2026\SIGMA\Matlab Codes';

DATASETS = ["TruthfulQA","SimpleQA","FreshQA"];
AR_PASS  = 0.90;        % gate calibrated to accept this fraction of AR clouds
GATE_SIG = 'hamzah';    % consensus signal (rho)

% pooled accumulators (logical column vectors)
P_gate = false(0,1);  P_rtr = false(0,1);
P_AW   = false(0,1);  P_AR  = false(0,1);  P_myth = false(0,1);

fprintf('%-12s %5s %9s %10s %9s %9s %8s %9s\n', ...
    'dataset','n_AW','FA_gate','FA_router','vload_g','vload_r','myth@AW','myth_all');
fprintf('%s\n', repmat('-',1,80));

for d = 1:numel(DATASETS)
    ds = DATASETS(d);
    S  = load(sprintf('stage4_by_state_%s.mat', ds));
    rho   = S.(GATE_SIG)(:);
    state = S.state(:);
    qids  = double(S.qids(:));

    % --- myth flags aligned to qids ---
    Mt = readtable(sprintf('%s_mythprone.xlsx', ds));
    mq = double(Mt.question_id);
    mf = double(Mt.myth_prone);
    [tf, loc] = ismember(qids, mq);
    myth = false(numel(qids),1);
    myth(tf) = mf(loc(tf)) > 0.5;          % unmatched -> false (not routed)
    if any(~tf)
        fprintf('  [%s] NOTE: %d qids lacked a myth flag (treated as not myth-prone)\n', ...
                ds, sum(~tf));
    end

    isAR = state == "all_right";
    isAW = state == "all_wrong";
    if sum(isAR) < 20
        fprintf('  [%s] NOTE: AR cell n=%d is small; tau is power-limited here.\n', ds, sum(isAR));
    end

    % --- gate threshold from AR only ---
    tau = quantile(rho(isAR), 1 - AR_PASS);

    accept_gate = rho >= tau;
    accept_rtr  = accept_gate & ~myth;

    n_AW    = sum(isAW);
    FA_g    = mean(accept_gate(isAW));
    FA_r    = mean(accept_rtr(isAW));
    vload_g = mean(~accept_gate);
    vload_r = mean(~accept_rtr);
    recAW   = mean(myth(isAW));
    rate    = mean(myth);

    fprintf('%-12s %5d %8.0f%% %9.0f%% %8.0f%% %8.0f%% %7.0f%% %8.0f%%\n', ...
        ds, n_AW, 100*FA_g, 100*FA_r, 100*vload_g, 100*vload_r, 100*recAW, 100*rate);

    % --- accumulate pooled (force logical column vectors) ---
    P_gate = [P_gate; logical(accept_gate(:))];
    P_rtr  = [P_rtr;  logical(accept_rtr(:))];
    P_AW   = [P_AW;   logical(isAW(:))];
    P_AR   = [P_AR;   logical(isAR(:))];
    P_myth = [P_myth; logical(myth(:))];
end

%% --- pooled summary ---
fprintf('\nPooled across datasets (n_AW=%d):\n', sum(P_AW));
fprintf('  AW false-accept    gate = %.0f%%   router = %.0f%%\n', ...
    100*mean(P_gate(P_AW)), 100*mean(P_rtr(P_AW)));
fprintf('  Verification load  gate = %.0f%%   router = %.0f%%\n', ...
    100*mean(~P_gate), 100*mean(~P_rtr));
extra = P_AR & P_gate & ~P_rtr;
fprintf('  Router over-verifies %.0f%% of AR (correct-consensus) clouds the gate would pass\n', ...
    100*sum(extra)/max(sum(P_AR),1));
