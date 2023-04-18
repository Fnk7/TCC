A = transpose(X) * inv(M) * X
b = inv(A) * transpose(X) * inv(M) * y

L = factorize(M)                // dpotrf
W = solve(L, X)                 // dtrsm
S = transpose(W) * W            // dsyrk
G = factorize(S)                // dpotrf
b = solve(L, y)                 // dtrsv
b = transpose(W) * b            // dgemv
b = solve(transpose(G), b)      // dtrsv
b = solve(G, b)                 // dtrsv

