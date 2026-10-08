% clear; clc; close all;

%% 1. 路徑
OUT = "C:\Lab\brain_age_test_retest\output";
PLOT_DIR = fullfile(OUT, "plots_MACL");

if ~isfolder(PLOT_DIR)
    mkdir(PLOT_DIR);
end

%% 2. 檔案
FILES = [
    fullfile(OUT, "ResNet_batch1.xlsx")
    fullfile(OUT, "SFCN_reg.xlsx")
    fullfile(OUT, "SFCN_sm.xlsx")
];

%% 3. Model 順序
CFG = {
    "SFCN-reg T1WI",    2, "Maclaren_T1WI";
    "SFCN-reg GM",      2, "Maclaren_GMP";
    "SFCN-reg WM",      2, "Maclaren_WMP";
    "SFCN-sm T1WI",     3, "Maclaren_T1WI";
    "SFCN-sm GM",       3, "Maclaren_GMP";
    "SFCN-sm WM",       3, "Maclaren_WMP";
    "ResNet T1",        1, "Maclaren_T1";
    "ResNet T1 B0",     1, "Maclaren_T1_B0";
    "ResNet T1 SEX",    1, "Maclaren_T1_SEX";
    "ResNet T1 SEX B0", 1, "Maclaren_T1_SEX_B0"
};

nModels = size(CFG,1);

%% 4. 顏色
SUB_COLORS = [
    0.35 0.80 0.45;   % Sub1
    0.35 0.60 1.00;   % Sub2
    0.95 0.45 0.90    % Sub3
];

MEAN_COLOR = [0.35 0.35 0.35];

USE_FIXED_YLIM = true;
COMMON_YLIM = [0 12];

%% 5. 儲存
diffData = cell(nModels,1);
subData  = cell(nModels,1);

Model    = strings(nModels,1);
Npairs   = zeros(nModels,1);

Sub1Mean = nan(nModels,1);
Sub2Mean = nan(nModels,1);
Sub3Mean = nan(nModels,1);

MeanDiff = nan(nModels,1);
MinDiff  = nan(nModels,1);
MaxDiff  = nan(nModels,1);

%% 6. 主迴圈
for m = 1:nModels

    Model(m) = CFG{m,1};
    filePath = FILES(CFG{m,2});
    sheetName = CFG{m,3};

    if ~isfile(filePath)
        warning("File not found: %s", filePath);
        continue;
    end

    if ~ismember(sheetName, string(sheetnames(filePath)))
        warning("Sheet not found: %s | %s", filePath, sheetName);
        continue;
    end

    %% 讀資料
    T = readtable( ...
        filePath, ...
        'Sheet', sheetName, ...
        'VariableNamingRule', 'preserve');

    T = T(:,1:3);
    T.Properties.VariableNames = ...
        {'SubjectID','TrueAge','PredictedAge'};

    T.PredictedAge = toNumeric(T.PredictedAge);

    ok = ...
        ~isnan(T.PredictedAge) & ...
        strlength(strtrim(string(T.SubjectID))) > 0;

    T = T(ok,:);

    %% 解析 subject / run
    [subID, runID] = parseID(string(T.SubjectID));

    allDiff = [];
    allSub  = [];

    subjectMeans = nan(3,1);

    %% 每位 subject
    for s = 1:3

        idx = (subID == s);

        subRun  = runID(idx);
        subPred = T.PredictedAge(idx);

        % Maclaren 應該每位 subject 都有 40 scans
        if numel(subRun) ~= 40
            error( ...
                "%s | Sub%d: expected 40 scans, found %d", ...
                Model(m), s, numel(subRun));
        end

        intraDiff = nan(20,1);

        %% 20 個 session
        for sess = 1:20

            expRuns = [2*sess-1, 2*sess];

            pairIdx = ismember(subRun, expRuns);

            runs = subRun(pairIdx);
            pred = subPred(pairIdx);

            %% 必須剛好兩張
            if numel(pred) ~= 2
                error( ...
                    "%s | Sub%d | Session%d: expected 2 scans, found %d", ...
                    Model(m), s, sess, numel(pred));
            end

            %% 必須是正確 run pair
            if ~isequal(sort(runs(:))', expRuns)
                error( ...
                    "%s | Sub%d | Session%d: expected runs [%d %d], found [%s]", ...
                    Model(m), s, sess, ...
                    expRuns(1), expRuns(2), ...
                    num2str(sort(runs(:))'));
            end

            %% 按 run 順序排列
            [~, order] = sort(runs);
            pred = pred(order);

            %% 同一天兩次 scan 相減
            intraDiff(sess) = abs(pred(1) - pred(2));
        end

        %% 每位 subject 應有 20 個 intra-session difference
        if sum(~isnan(intraDiff)) ~= 20
            error( ...
                "%s | Sub%d: expected 20 intra-session pairs.", ...
                Model(m), s);
        end

        subjectMeans(s) = mean(intraDiff);

        allDiff = [allDiff; intraDiff]; %#ok<AGROW>
        allSub  = [allSub; repmat(s,20,1)]; %#ok<AGROW>
    end

    %% 儲存
    diffData{m} = allDiff;
    subData{m}  = allSub;

    % 20 pairs × 3 subjects = 60
    Npairs(m) = numel(allDiff);

    if Npairs(m) ~= 60
        error( ...
            "%s: expected 60 intra-session pairs, found %d", ...
            Model(m), Npairs(m));
    end

    Sub1Mean(m) = subjectMeans(1);
    Sub2Mean(m) = subjectMeans(2);
    Sub3Mean(m) = subjectMeans(3);

    MeanDiff(m) = mean(allDiff);
    MinDiff(m)  = min(allDiff);
    MaxDiff(m)  = max(allDiff);
end

%% 7. Summary
Summary = table( ...
    Model, Npairs, ...
    Sub1Mean, Sub2Mean, Sub3Mean, ...
    MeanDiff, MinDiff, MaxDiff);

fprintf("\n========== Maclaren intra-session summary ==========\n");
disp(Summary);

csvPath = fullfile( ...
    PLOT_DIR, ...
    "MACL_IntraSession_Summary.csv");

writetable(Summary, csvPath);

%% 8. 畫圖
f = figure( ...
    'Color','w', ...
    'Position',[100 100 1650 720]);

hold on;

SUB_OFFSET = [-0.13, 0, 0.13];
rng(1);

for m = 1:nModels

    y   = diffData{m};
    sub = subData{m};

    if isempty(y)
        continue;
    end

    for s = 1:3

        idx = (sub == s);

        x = ...
            m + SUB_OFFSET(s) + ...
            (rand(sum(idx),1)-0.5)*0.10;

        scatter( ...
            x, y(idx), 38, 'o', ...
            'MarkerFaceColor', SUB_COLORS(s,:), ...
            'MarkerEdgeColor', 'none', ...
            'MarkerFaceAlpha', 0.65);
    end

    %% 60 點總平均
    plot( ...
        [m-0.25, m+0.25], ...
        [MeanDiff(m), MeanDiff(m)], ...
        '--', ...
        'Color', MEAN_COLOR, ...
        'LineWidth', 2);

    text( ...
        m, MeanDiff(m)+0.5, ...
        sprintf('%.2f', MeanDiff(m)), ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom', ...
        'FontSize',11, ...
        'FontWeight','bold', ...
        'Color',MEAN_COLOR);
end

%% 9. 圖形格式
xlim([0.45, nModels+0.55]);

xticks(1:nModels);
xticklabels(Model);
xtickangle(35);

xlabel('Model', 'FontSize',15);
ylabel('|Scan 1 - Scan 2| (years)', 'FontSize',15);
title('Maclaren', 'FontSize',20, 'FontWeight','bold');

if USE_FIXED_YLIM
    ylim(COMMON_YLIM);
    yticks(COMMON_YLIM(1):1:COMMON_YLIM(2));
else
    allY = vertcat(diffData{:});
    yTop = max(allY, [], 'omitnan');
    ylim([0, max(1, ceil(yTop*1.12))]);
end

set(gca, ...
    'FontSize',12, ...
    'LineWidth',1, ...
    'TickDir','out');

grid on;
box on;

%% 10. Legend
h1 = scatter(nan,nan,55,'o','filled', ...
    'MarkerFaceColor',SUB_COLORS(1,:), ...
    'MarkerEdgeColor','none');

h2 = scatter(nan,nan,55,'o','filled', ...
    'MarkerFaceColor',SUB_COLORS(2,:), ...
    'MarkerEdgeColor','none');

h3 = scatter(nan,nan,55,'o','filled', ...
    'MarkerFaceColor',SUB_COLORS(3,:), ...
    'MarkerEdgeColor','none');

hMean = plot( ...
    nan,nan,'--', ...
    'Color',MEAN_COLOR, ...
    'LineWidth',2);

legend( ...
    [h1,h2,h3,hMean], ...
    {'Sub 1','Sub 2','Sub 3','Mean'}, ...
    'Location','northeast');

%% 11. Save
figPath = fullfile( ...
    PLOT_DIR, ...
    "MACL_IntraSession_Distribution.png");

exportgraphics(f, figPath, 'Resolution',300);

fprintf("Figure: %s\n", figPath);
fprintf("Table : %s\n", csvPath);

%% ================================================================
% Functions
% ================================================================
function [subID, runID] = parseID(ids)

    n = numel(ids);

    subID = nan(n,1);
    runID = nan(n,1);

    for i = 1:n

        tok = regexp( ...
            char(ids(i)), ...
            'sub[-_]?0*(\d+).*run[-_]?0*(\d+)', ...
            'tokens', ...
            'once');

        if isempty(tok)
            error( ...
                "Cannot parse Subject/Run from Subject ID: %s", ...
                ids(i));
        end

        subID(i) = str2double(tok{1});
        runID(i) = str2double(tok{2});
    end
end

function x = toNumeric(x)

    if isnumeric(x)
        x = double(x);
    else
        x = str2double(string(x));
    end
end