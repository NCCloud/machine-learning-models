function D = processData(file)
    A = csvread(file);
    size(A)
    D = A(:, 2:13);             % reduce columns by removing user_cancelled values 
                                % and the subscription id columns
    D = D(2:end, :);            % remove header row
    size(D)
    T = D(10001:10100, :);      % test dataset of 100 rows
    
    X = D(20001:20002, :);          % reduce number of rows
    D = D(1:10000, :);          % reduce number of rows
    X
    size(D)
    size(T)

    % Normalize dataset
    for i = 1:size(D)(1,2)
        D(:, i) = normalize(D(:, i));
    end
    % Normalize Test dataset too
    for i = 1:size(T)(1,2)
        T(:, i) = normalize(T(:, i));
    end

    % Save matrices as csv files
    csvwrite(cstrcat(
        file(1:size(file)(1,2) - 4), '_norm', '.csv'
    ), D);

    csvwrite(cstrcat(
        file(1:size(file)(1,2) - 4), '_norm_test', '.csv'
    ), T);
