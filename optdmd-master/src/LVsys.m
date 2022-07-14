function dydt = LVsys(t,y,b,p,r,d)
dydt = [(b-p*y(2))*y(1); (r*y(1)-d)*y(2)];