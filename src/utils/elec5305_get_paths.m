function P = elec5305_get_paths(startHere)
% Robust project path resolver (works in Live Editor & different machines)
% P has fields: root, src, results, audio, plots, ml, modelFile

if nargin<1 || isempty(startHere)
    % Try active editor file; fallback to pwd
    try
        f = matlab.desktop.editor.getActiveFilename;
        if ~isempty(f), startHere = fileparts(f); end
    catch
        startHere = pwd;
    end
end

% Walk upwards to find a folder that has src/ and results/ (or a .git)
root = startHere;
for k = 1:10
    hasGit = isfolder(fullfile(root,'.git'));
    hasSR  = isfolder(fullfile(root,'src')) && isfolder(fullfile(root,'results'));
    if hasGit || hasSR
        break;
    end
    parent = fileparts(root);
    if strcmp(parent,root)
        error(['[paths] Cannot locate project root from: ' startHere ...
               ' (expecting a folder containing src/ and results/)']);
    end
    root = parent;
end

P.root    = root;
P.src     = fullfile(root,'src');
P.results = fullfile(root,'results');
P.audio   = fullfile(P.results,'audio');
P.plots   = fullfile(P.results,'plots');
P.ml      = fullfile(P.results,'ml');
P.modelFile = fullfile(P.ml,'instrument_classifier.mat');
end
