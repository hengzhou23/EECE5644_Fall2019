function plot_scatter(data)
    labels = data(:,3);
    gscatter(data(:,1), data(:,2), labels, 'rk', 'ox');
    xlabel('Data 1');
    ylabel('Data 2');
    axis([-4 4 -4 4]);
    grid on;
end