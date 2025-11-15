function setup()
% setup.m — ELEC5305 project bootstrap (no absolute paths)
% Usage: run once after cloning the repo.

paths = elec5305_get_paths();

% Add source code to path
addpath(genpath(paths.src));

% Ensure results subfolders exist
cellfun(@(d) ~isfolder(d) && mkdir(d), ...
    {paths.results, paths.audio, paths.plots, paths.ml});

fprintf('[setup] Project root : %s\n', paths.root);
fprintf('[setup] Added to path: %s\n', paths.src);
fprintf('[setup] Results dir  : %s\n', paths.results);
end
