function fitness = LVopt(x)

global xdata

b=x(1);
p=x(2);
r=x(3);
d=x(4);

y0=[32000;20000];
tspan=0:2:58;
[tode,y] = ode45(@(t,y) LVsys(t,y,b,p,r,d), tspan, y0);
V=r*xdata(1,:)-d*log(xdata(1,:))+p*xdata(2,:)-b*log(xdata(2,:));
Vmean=mean(V);
fitness=norm(y'-xdata,2)+norm(V-Vmean,2);