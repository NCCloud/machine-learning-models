% plot dataset dimensions
function plotDataset(X)
    t=[1:1:size(X)(1,1)];
    for i = 1:size(X)(1,2)
        subplot(4,3,i);
        plot(t, normalize(X(:, i)), '.');
    end