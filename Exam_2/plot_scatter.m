function plot_scatter(data)
    gscatter(data(:,1), data(:,2), data(:,3), 'rk', 'ox');
    xlabel('Data 1');
    ylabel('Data 2');
    axis([-4 4 -4 4]);
    grid on;
end