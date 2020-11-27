function N = normalize(X)
    N = X(:) / max(X(:));