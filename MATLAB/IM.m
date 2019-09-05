%Induction Motor

function [Ir, Vr, Te, efficiency, Is, cosPhi, Po] = InductionMotor(s)
  Wms = 157.079; Ns = 1500;
  Ir_ = (223.2948301+4.638370655i)/(9.63i+1.55+(0.8/s));
  Te = (3/(s*Wms))*((abs(Ir_)^2)*0.8)
  Ia = ((4.967i+(0.8/s))/(2.969+69.37i))*Ir_
  Is = Ia+Ir_
  cosPhi = real(Is)/abs(Is)
  Po = Te*(1-s)*Wms;
  Pi = sqrt(3)*415*real(Is);
  efficiency = Po/Pi
  Ir = Ir_*3.6
  Vr = Ir_*sqrt(3)*(4.967i+(0.8/s))*(3.6**-1)
end

#Gathering values for plotting for given precision

s = 0;
Ir_pts = [];
Vr_pts = [];
Te_pts = [];
efficiency_pts = [];
Is_pts = [];
cosPhi_pts = [];
Po_pts = [];
S = [];
precision = 0.01;
while(s<=1)
  s = s+precision;
  S = [S; s];
  [Ir, Vr, Te, efficiency, Is, cosPhi, Po] = InductionMotor(s);
  Ir_pts = [Ir_pts; Ir]
  Vr_pts = [Vr_pts; Vr]
  Te_pts = [Te_pts; Te]
  efficiency_pts = [efficiency_pts; efficiency]
  Is_pts = [Is_pts; Is]
  cosPhi_pts = [cosPhi_pts; cosPhi]
  Po_pts = [Po_pts; Po]
end

figure(1);
plot(Po_pts, S)
xlabel("Po")
ylabel("Slip")
title("slip vs Po")
print -dpng "slipVsPo.png"
figure(2);
plot(Po_pts, Te_pts)
xlabel("PO")
ylabel("Torque")
title("Torque(Te) vs Po")
print -dpng "TorqueVsPo.png"
figure(3);
plot(Po_pts, Is_pts)
xlabel("Po")
ylabel("Is")
title("Is vs Po")
print -dpng "IsVsPo.png"
figure(4);
plot(Po_pts, efficiency_pts)
xlabel("Po")
ylabel("efficiency")
title("efficiency vs Po")
print -dpng "efficiencyVsPo.png"
figure(5);
plot(Po_pts, Ir_pts)
xlabel("Po")
ylabel("Ir")
title("Ir vs Po")
print -dpng "IrVsPo.png"
figure(6);
plot(Po_pts, cosPhi_pts)
xlabel("Po")
ylabel("cosPhi")
title("cosPhi vs Po")
print -dpng "cosPhiVsPo.png"
figure(7);
plot(Po_pts, Vr_pts)
xlabel("Po")
ylabel("Vr")
title("Vr vs Po")
print -dpng "VrVsPo.png"
figure(8);
plot(S, Te_pts)
xlabel("Slip")
ylabel("Torque")
title("Torque vs Slip")
print -dpng "TorqueVsSlip.png"