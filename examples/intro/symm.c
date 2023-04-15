double X, Y, Z, A, B, C, D;

void foH() {

B = dgemm(transpose(X), X);
C = dgemm(transpose(X), Y);
A = B + C + transpose(C);
// B = dsyrk(X)

Z = 0.5 * X + Y;
D = dgemm(transpose(X), Z);
A = D + transpose(D);
// A = dsyr2k(X, Z)


}

