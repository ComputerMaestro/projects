N = input('Enter number of buses:\n');
Nlines = input('Enter number of lines:\n');
Ybus = zeros(N, N);
for line=1:Nlines
    fprintf('Enter values for line %d\n', line);
    p = input('Enter "from bus code" for line:\n');
    q = input('Enter "to bus code" for line:\');
    r = input('Enter resistance of line in p.u.:\n');
    x = input('Enter reactance of line in p.u.:\n');
    a = input('Enter half line charging admittance in p.u.:\n');
    y = 1/(r+i*x);
    Ybus(p, q) = Ybus(p, q) - y;
    Ybus(q, p) = Ybus(q, p) - y;
    Ybus(p, p) = Ybus(p, p) + y + i*a;
    Ybus(q, q) = Ybus(q, q) + y + i*a;
end
disp('Y bus matrix is:\n')
disp(Ybus)