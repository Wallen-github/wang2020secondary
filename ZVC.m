clear all
rs = 163.0/1.2;
aA = 1.44*rs; 
bA = 1.2*rs; 
cA = 1*rs; 
Rp = 780.0; 
miu = aA*bA*cA/(Rp^3+aA*bA*cA); 
a = aA+Rp; 
m = miu*(1-miu); 
a_bar = (aA*bA*cA)^(1/3); 
alpha = a_bar/a; 
J2 = (aA^2+bA^2-2*cA^2)/(10*a_bar^2); 
J22 = (aA^2-bA^2)/(20*a_bar^2); 
A = alpha^2*J2; 
B = 6*alpha^2*J22; 
IzA = miu*(aA^2+bA^2)/(5*a^2);
i = 1;
j = 1;
ii = 1;
jj = 1;
K = 0.01;
E = -0.01;
flag = 0;
for x = -2:0.005:2
    for y = -2:0.005:2
        if sqrt(x^2+y^2)>1
            S = sqrt(x^2+y^2);
            theta = atan2(y,x);
            ZVCs(i,j) = E-K^2/(2*(IzA+m*S^2))+m/S+3*m/(2*S^3)*(A+B*cos(2*theta));
            if ZVCs(i,j)>-0.0066
                Z(i,j) = ZVCs(i,j);
            else
                Z(i,j) = nan;
            end
        else
            ZVCs(i,j)= nan;
        end
        j = j+1;
    end
    j = 1;
    i = i+1;
end

Z = Z';
contour([-2:0.005:2],[-2:0.005:2],Z,1000);

hold on

grid on 

t=[0:0.1:2*pi];

xx=aA/(aA+Rp)*cos(t);

yy=bA/(aA+Rp)*sin(t);

plot(xx,yy);

fill(xx,yy,'black');

xlabel('x');% x÷·√˚≥∆
ylabel('y'); 
c = colorbar;


