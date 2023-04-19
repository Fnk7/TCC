B = dgemm(transpose(X), X);
# B = dsyrk(X)
C = dgemm(transpose(X), Y);
H = B + C + transpose(C);

Z = 0.5 * X + Y;
D = dgemm(transpose(X), Z);
H = D + transpose(D);
# H = dsyr2k(X, Z)


