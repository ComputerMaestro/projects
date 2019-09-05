base_V = input('Enter base voltage:\n');
base_MVA = input('Enter base MVA value:\n');
base_imped = ((base_V/1000)^2)/base_MVA
q = 0
while q != 1
  V = input('Enter voltage to calculate the per unit voltage:\n');
  V = V/1000;
  z = input('Enter impedance to calculate the per unit value:\n');
  disp('per unit voltage: ');disp(V/base_V);
  disp('per unit impedance: ');disp(z/base_imped);
  q = input('press "q" to stop calculating per unit values, else press any other key', "s") == 'q';
endwhile
N = input('Enter number of buses:\n');
B = input('Enter number of branches:\n');
Z = zeros(N, N);
for p=1:N
  for q=1:N
    Z(p, q) = input('');
  endfor
endfor
n = N+1;
bus = input('Enter bus code to connect to ground:\n');
z = input('Enter impedence of the new branch:\n');
Z = [Z , Z(:, bus)];
Z = [Z ; [Z(bus, 1:end-1), z+Z(bus, bus)]]
for p=1:N
  for q=1:N
    Z(p, q) = Z(p, q) - ((Z(p, n)*Z(n, q))/(Z(n, n)));
  endfor
endfor
Z = Z(1:N, 1:N);
disp(Z)
