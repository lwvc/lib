clear; clc; close all;

%% =========================================================
% File paths
% =========================================================
baseDir = 'C:\Lab\brain_age_test_retest\output';

resnetFile  = fullfile(baseDir, 'ResNet.xlsx');
sfcnRegFile = fullfile(baseDir, 'SFCN_reg.xlsx');
sfcnSmFile  = fullfile(baseDir, 'SFCN_sm.xlsx');

%% =========================================================
% OAS1 overlap subjects (remove only for ResNet)
% =========================================================
OAS1_RESNET_OVERLAP = [
    "OAS1_0061"
    "OAS1_0092"
    "OAS1_0111"
    "OAS1_0150"
    "OAS1_0156"
    "OAS1_0202"
    "OAS1_0230"
    "OAS1_0353"
    "OAS1_0368"
    "OAS1_0379"
];

%% =========================================================
% Model colors
% =========================================================
MODEL_COLORS = [
    0.00 0.45 0.15;   % SFCN-reg T1WI
    0.20 0.65 0.25;   % SFCN-reg GM
    0.55 0.82 0.55;   % SFCN-reg WM
    0.70 0.00 0.00;   % SFCN-sm T1WI
    0.88 0.28 0.28;   % SFCN-sm GM
    0.97 0.65 0.65;   % SFCN-sm WM
    0.00 0.25 0.75;   % ResNet T1
    0.15 0.45 0.90;   % ResNet T1 B0
    0.40 0.65 0.95;   % ResNet T1 SEX
    0.72 0.84 0.98;   % ResNet T1 SEX B0
];

%% =========================================================
% Model configuration
% name | family | file | MMRR sheet | OAS1 sheet | Maclaren sheet
% =========================================================
models = {
    'SFCN-reg T1WI',   'sfcn_reg', sfcnRegFile, 'MMRR_T1WI',      'OAS1_T1WI',      'Maclaren_T1WI'
    'SFCN-reg GM',     'sfcn_reg', sfcnRegFile, 'MMRR_GMP',       'OAS1_GMP',       'Maclaren_GMP'
    'SFCN-reg WM',     'sfcn_reg', sfcnRegFile, 'MMRR_WMP',       'OAS1_WMP',       'Maclaren_WMP'

    'SFCN-sm T1WI',    'sfcn_sm',  sfcnSmFile,  'MMRR_T1WI',      'OAS1_T1WI',      'Maclaren_T1WI'
    'SFCN-sm GM',      'sfcn_sm',  sfcnSmFile,  'MMRR_GMP',       'OAS1_GMP',       'Maclaren_GMP'
    'SFCN-sm WM',      'sfcn_sm',  sfcnSmFile,  'MMRR_WMP',       'OAS1_WMP',       'Maclaren_WMP'

    'ResNet T1',       'resnet',   resnetFile,  'MMRR_T1',        'OAS1_T1',        'Maclaren_T1'
    'ResNet T1 B0',    'resnet',   resnetFile,  'MMRR_T1_B0',     'OAS1_T1_B0',     'Maclaren_T1_B0'
    'ResNet T1 SEX',   'resnet',   resnetFile,  'MMRR_T1_SEX',    'OAS1_T1_SEX',    'Maclaren_T1_SEX'
    'ResNet T1 SEX B0','resnet',   resnetFile,  'MMRR_T1_SEX_B0', 'OAS1_T1_SEX_B0', 'Maclaren_T1_SEX_B0'
};

nModels = size(models,1);

%% =========================================================
% Containers
% =========================================================
MMRR_MAE      = nan(nModels,1);
MMRR_MAD      = nan(nModels,1);

OAS1_MAE      = nan(nModels,1);
OAS1_MAD      = nan(nModels,1);

Maclaren_MAE  = nan(nModels,1);
Maclaren_MAD  = nan(nModels,1);   % this is MADintra

%% =========================================================
% Compute metrics
% =========================================================
for k = 1:nModels

    modelName = models{k,1};
    family    = models{k,2};
    fileName  = models{k,3};
    mmrrSheet = models{k,4};
    oas1Sheet = models{k,5};
    maclSheet = models{k,6};

    fprintf('\n====================================\n');
    fprintf('%s\n', modelName);
    fprintf('====================================\n');

    %% ---------------- MMRR ----------------
    T = readtable(fileName, 'Sheet', mmrrSheet, ...
        'VariableNamingRule','preserve');

    ID   = string(T{:,1});
    age  = double(T{:,2});
    pred = double(T{:,3});

    subj = extractMMRRSubject(ID);

    %% MMRR MAD
    % MAD 不使用 chronological age，因此保留全部 21 pairs
    Y = makePairsBySubject(subj, ID, pred);
    
    MMRR_MAD(k) = mean( ...
        abs(Y(:,1) - Y(:,2)), ...
        'omitnan');
    
    %% MMRR MAE
    % 排除 sub_502 與 sub_742
    % 兩位各有 2 scans，共排除 4 scans
    
    MMRR_EXCLUDE = [
        "sub_502"
        "sub_742"
    ];
    
    keep_MAE = ~ismember(subj, MMRR_EXCLUDE);
    
    age_MAE  = age(keep_MAE);
    pred_MAE = pred(keep_MAE);
    
    MMRR_MAE(k) = mean( ...
        abs(pred_MAE - age_MAE), ...
        'omitnan');

    %% ---------------- OAS1 ----------------
    T = readtable(fileName, 'Sheet', oas1Sheet, ...
        'VariableNamingRule','preserve');

    ID   = string(T{:,1});
    age  = double(T{:,2});
    pred = double(T{:,3});

    subj = extractOAS1Subject(ID);

    % Remove overlap only for ResNet
    if strcmpi(family, 'resnet')
        keep = ~ismember(subj, OAS1_RESNET_OVERLAP);
        ID   = ID(keep);
        age  = age(keep);
        pred = pred(keep);
        subj = subj(keep);
    end

    Y = makePairsBySubject(subj, ID, pred);

    OAS1_MAE(k) = mean(abs(pred - age), 'omitnan');
    OAS1_MAD(k) = mean(abs(Y(:,1) - Y(:,2)), 'omitnan');

    %% ---------------- Maclaren ----------------
    T = readtable(fileName, 'Sheet', maclSheet, ...
        'VariableNamingRule','preserve');

    ID   = string(T{:,1});
    age  = double(T{:,2});
    pred = double(T{:,3});

    Maclaren_MAE(k) = mean(abs(pred - age), 'omitnan');
    Maclaren_MAD(k) = computeMaclarenMADintra(ID, pred);

    fprintf('MMRR      MAE = %.3f, MAD = %.3f\n', MMRR_MAE(k), MMRR_MAD(k));
    fprintf('OAS1      MAE = %.3f, MAD = %.3f\n', OAS1_MAE(k), OAS1_MAD(k));
    fprintf('Maclaren  MAE = %.3f, MAD = %.3f\n', Maclaren_MAE(k), Maclaren_MAD(k));
end

%% =========================================================
% Plot: MAE vs MAD
% =========================================================
figure('Color','w','Position',[80 120 1550 420]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

datasetTitles = {'MMRR : MAE vs MAD', 'OAS1 : MAE vs MAD', 'Maclaren : MAE vs MAD'};

X_all = {MMRR_MAE, OAS1_MAE, Maclaren_MAE};
Y_all = {MMRR_MAD, OAS1_MAD, Maclaren_MAD};

for d = 1:3
    nexttile;
    hold on;

    for k = 1:nModels
        scatter(X_all{d}(k), Y_all{d}(k), 78, ...
            'o', ...
            'MarkerFaceColor', MODEL_COLORS(k,:), ...
            'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.8, ...
            'DisplayName', models{k,1});
    end

    xlabel('MAE (years)', 'FontSize', 11);
    ylabel('MAD (years)', 'FontSize', 11);
    title(datasetTitles{d}, 'FontSize', 12, 'FontWeight', 'bold');

    xlim([0 11]);
    ylim([0 3]);

    grid on;
    box on;
    set(gca, 'FontSize', 10, 'LineWidth', 1);

    if d == 1
        lgd = legend('Location','northwest', 'FontSize', 9);
        lgd.Box = 'on';
    end

    hold off;
end

%% =========================================================
% Save figure
% =========================================================
saveas(gcf, fullfile(baseDir, 'MAE_vs_MAD_all_datasets.png'));

%% =========================================================
% Save results table
% =========================================================
Result = table( ...
    string(models(:,1)), ...
    MMRR_MAE, MMRR_MAD, ...
    OAS1_MAE, OAS1_MAD, ...
    Maclaren_MAE, Maclaren_MAD, ...
    'VariableNames', { ...
    'Model', ...
    'MMRR_MAE', 'MMRR_MAD', ...
    'OAS1_MAE', 'OAS1_MAD', ...
    'Maclaren_MAE', 'Maclaren_MAD'});

disp(Result);

writetable(Result, fullfile(baseDir, 'MAE_vs_MAD_values.xlsx'));

%% =========================================================
% Local functions
% =========================================================

function subj = extractMMRRSubject(ID)
    subj = strings(size(ID));
    for i = 1:length(ID)
        token = regexp(ID(i), '^(sub_\d+)', 'tokens', 'once');
        if ~isempty(token)
            subj(i) = string(token{1});
        end
    end
end

function subj = extractOAS1Subject(ID)
    subj = strings(size(ID));
    for i = 1:length(ID)
        token = regexp(ID(i), '^(OAS1_\d+)', 'tokens', 'once');
        if ~isempty(token)
            subj(i) = string(token{1});
        end
    end
end

function Y = makePairsBySubject(subj, ID, pred)
    uSubj = unique(subj);
    uSubj(uSubj=="") = [];

    Y = [];

    for s = 1:length(uSubj)
        idx = find(subj == uSubj(s));

        if numel(idx) ~= 2
            warning('%s has %d scans, expected 2.', uSubj(s), numel(idx));
            continue;
        end

        [~, ord] = sort(ID(idx));
        idx = idx(ord);

        Y(end+1,:) = [pred(idx(1)), pred(idx(2))]; %#ok<AGROW>
    end
end

function MADintra = computeMaclarenMADintra(ID, pred)
    % Expect IDs containing sub-01 / sub-02 / sub-03 and run-01 ... run-40

    subj = strings(size(ID));
    runNum = nan(size(ID));

    for i = 1:length(ID)
        tokSub = regexp(ID(i), '(sub-\d+)', 'tokens', 'once');
        if ~isempty(tokSub)
            subj(i) = string(tokSub{1});
        end

        tokRun = regexp(ID(i), 'run-(\d+)', 'tokens', 'once');
        if ~isempty(tokRun)
            runNum(i) = str2double(tokRun{1});
        end
    end

    uSubj = unique(subj);
    uSubj(uSubj=="") = [];

    allDiff = [];

    for s = 1:length(uSubj)
        idx = find(subj == uSubj(s));
        [runSorted, ord] = sort(runNum(idx));
        predSorted = pred(idx);
        predSorted = predSorted(ord);

        if numel(predSorted) ~= 40
            warning('%s has %d scans, expected 40.', uSubj(s), numel(predSorted));
            continue;
        end

        if ~isequal(runSorted(:)', 1:40)
            warning('%s run numbers are incomplete.', uSubj(s));
            continue;
        end

        scan1 = predSorted(1:2:end);
        scan2 = predSorted(2:2:end);

        allDiff = [allDiff; abs(scan1 - scan2)]; %#ok<AGROW>
    end

    MADintra = mean(allDiff, 'omitnan');
end