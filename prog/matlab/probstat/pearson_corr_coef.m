function res = pearson_corr_coef(x, y)
    % Pearson correlation coefficient of two samples
    dx = x - mean(x);
    dy = y - mean(y);
    nume = sum(dx .* dy);
    deno = sqrt(sum(dx.^2)) * sqrt(sum(dy.^2));
    res = nume/deno;
end