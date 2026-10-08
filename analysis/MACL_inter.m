% clear; clc; close all;

%% 1. 路徑
OUT = "C:\Lab\brain_age_test_retest\output";
PLOT_DIR = "C:\Lab\brain_age_test_retest\output\plots_MACL";

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
% 顯示名稱使用 GM / WM；Excel 活頁名稱依目前檔案內容保留 GMP / WMP。
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

%% 4. Subject 顏色
SUB_COLORS = [
    0.35 0.80 0.45;   % Sub 1
    0.35 0.60 1.00;   % Sub 2
    0.95 0.45 0.90    % Sub 3
];

MAD_COLOR = [0.35 0.35 0.35];
SPECIAL_COLOR = [1.00 0.65 0.05];   % sub-03 run39/40
SPECIAL_SUB = 3;
SPECIAL_RUNS = [39 40];
SPECIAL_DAY = ceil(SPECIAL_RUNS(1)/2);

%% 5. 固定 y 軸
USE_FIXED_YLIM = true;
COMMON_YLIM = [0 12];

%% 6. 收集 inter-session deviation
deviationData = cell(nModels,1);
subjectData   = cell(nModels,1);
specialData   = cell(nModels,1);

Model   = strings(nModels,1);
N       = zeros(nModels,1);
MAD     = nan(nModels,1);
MinDev  = nan(nModels,1);
MaxDev  = nan(nModels,1);

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

    T = readtable( ...
        filePath, ...
        'Sheet', sheetName, ...
        'VariableNamingRule', 'preserve');

    T = cleanData(T);

    % Maclaren 每位 subject 有 40 runs；每 2 runs 為同一天的一個 session。
    [subID, runID] = parseID(string(T.SubjectID));
    dayID = ceil(runID / 2);

    allDeviation = [];
    allSubject   = [];
    allSpecial   = [];

    for s = 1:3
        idx = subID == s;

        subDay  = dayID(idx);
        subPred = T.PredictedAge(idx);

        validDays = unique(subDay(~isnan(subDay)));
        sessionMean = nan(numel(validDays),1);

        % 每個 session 先平均同一天的兩次 scan
        for d = 1:numel(validDays)
            dayIdx = subDay == validDays(d);
            pred = subPred(dayIdx);

            if numel(pred) ~= 2
                warning( ...
                    "%s | Sub %d | Day %g: expected 2 scans, found %d", ...
                    Model(m), s, validDays(d), numel(pred));
                continue;
            end

            sessionMean(d) = mean(pred, 'omitnan');
        end

        validMask = ~isnan(sessionMean);
        sessionMean = sessionMean(validMask);
        validDays = validDays(validMask);

        if isempty(sessionMean)
            continue;
        end

        % Subject 的 20-session 平均 predicted age
        subjectMean = mean(sessionMean, 'omitnan');

        % 每個 session 相對於該 subject 長期平均的 absolute deviation
        deviation = abs(sessionMean - subjectMean);

        allDeviation = [allDeviation; deviation]; %#ok<AGROW>
        allSubject = [allSubject; repmat(s, numel(deviation), 1)]; %#ok<AGROW>

        % 標記 sub-03 的 run39/40 所在 session（Day 20）
        isSpecial = (s == SPECIAL_SUB) & (validDays == SPECIAL_DAY);
        allSpecial = [allSpecial; isSpecial(:)]; %#ok<AGROW>
    end

    deviationData{m} = allDeviation;
    subjectData{m}   = allSubject;
    specialData{m}   = logical(allSpecial);

    N(m) = numel(allDeviation);

    if ~isempty(allDeviation)
        MAD(m)    = mean(allDeviation, 'omitnan');
        MinDev(m) = min(allDeviation, [], 'omitnan');
        MaxDev(m) = max(allDeviation, [], 'omitnan');
    end
end

%% 7. 畫圖
f = figure( ...
    'Color','w', ...
    'Position',[100 100 1650 720]);
hold on;

SUB_OFFSET = [-0.13, 0, 0.13];
rng(1);

for m = 1:nModels
    y   = deviationData{m};
    sub = subjectData{m};
    isSpecial = specialData{m};

    if isempty(y)
        continue;
    end

    for s = 1:3
        idx = (sub == s) & ~isSpecial;

        if ~any(idx)
            continue;
        end

        x = m + SUB_OFFSET(s) + ...
            (rand(sum(idx),1)-0.5) * 0.10;

        scatter( ...
            x, y(idx), 38, 'o', ...
            'MarkerFaceColor', SUB_COLORS(s,:), ...
            'MarkerEdgeColor', 'none', ...
            'MarkerFaceAlpha', 0.65);
    end

    % 特別標示 sub-03 run39/40
    if any(isSpecial)
        xSpecial = m + SUB_OFFSET(SPECIAL_SUB);
        scatter( ...
            xSpecial, y(isSpecial), 150, 'p', ...
            'MarkerFaceColor', SPECIAL_COLOR, ...
            'MarkerEdgeColor', [0.35 0.35 0.35], ...
            'LineWidth', 1.2);
    end

    plot( ...
        [m-0.25, m+0.25], ...
        [MAD(m), MAD(m)], ...
        '--', ...
        'Color', MAD_COLOR, ...
        'LineWidth', 2.0);

    text( ...
        m, ...
        MAD(m) + 0.5, ...
        sprintf('%.2f', MAD(m)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 11, ...
        'FontWeight', 'bold', ...
        'Color', MAD_COLOR);
end

%% 8. 圖形格式
xlim([0.45, nModels + 0.55]);
xticks(1:nModels);
xticklabels(Model);
xtickangle(35);

xlabel('Model', 'FontSize', 15);
ylabel('Inter-session deviation (years)', 'FontSize', 15);
title('Maclaren', 'FontSize', 20, 'FontWeight', 'bold');

if USE_FIXED_YLIM
    ylim(COMMON_YLIM);
    yticks(COMMON_YLIM(1):1:COMMON_YLIM(2));
else
    allY = vertcat(deviationData{:});
    yTop = max(allY, [], 'omitnan');
    ylim([0, max(1, ceil(yTop * 1.12))]);
end

set(gca, ...
    'FontSize', 12, ...
    'LineWidth', 1.0, ...
    'TickDir', 'out');

grid on;
box on;

%% 9. Legend
h1 = scatter(nan,nan,55,'o','filled', ...
    'MarkerFaceColor',SUB_COLORS(1,:), ...
    'MarkerEdgeColor','none');

h2 = scatter(nan,nan,55,'o','filled', ...
    'MarkerFaceColor',SUB_COLORS(2,:), ...
    'MarkerEdgeColor','none');

h3 = scatter(nan,nan,55,'o','filled', ...
    'MarkerFaceColor',SUB_COLORS(3,:), ...
    'MarkerEdgeColor','none');

hMAD = plot(nan,nan,'--', ...
    'Color',MAD_COLOR, ...
    'LineWidth',2.0);

hSpecial = scatter(nan,nan,150,'p','filled', ...
    'MarkerFaceColor',SPECIAL_COLOR, ...
    'MarkerEdgeColor',[0.35 0.35 0.35], ...
    'LineWidth',1.2);

legend( ...
    [h1,h2,h3,hMAD,hSpecial], ...
    {'Sub 1','Sub 2','Sub 3','MAD','sub-03 run39/40'}, ...
    'Location','northeast');

%% 10. 儲存
figPath = fullfile( ...
    PLOT_DIR, ...
    "MACL_InterSession_Deviation_Distribution.png");

exportgraphics(f, figPath, 'Resolution',300);

Summary = table(Model, N, MAD, MinDev, MaxDev);

csvPath = fullfile( ...
    PLOT_DIR, ...
    "MACL_InterSession_Deviation_Summary.csv");

writetable(Summary, csvPath);

fprintf("\n========== Maclaren inter-session summary ==========\n");
disp(Summary);
fprintf("圖：%s\n", figPath);
fprintf("表：%s\n", csvPath);

%% ================================================================
% Functions
% ================================================================
function T = cleanData(T)
    if width(T) < 3
        error("Excel must contain at least 4 columns.");
    end

    % 目前 Excel 欄位順序：
    % Subject ID | True Age | Predicted Age | File Name
    T = T(:,1:3);
    T.Properties.VariableNames = { ...
        'SubjectID', ...
        'TrueAge', ...
        'PredictedAge'};

    T.TrueAge      = toNumeric(T.TrueAge);
    T.PredictedAge = toNumeric(T.PredictedAge);

    id = strtrim(string(T.SubjectID));

    ok = ...
        ~isnan(T.PredictedAge) & ...
        strlength(id) > 0;

    T = T(ok,:);
end

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
