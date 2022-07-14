function dydt = LVsysSINDy(t,y,Xi)
dydt = [Xi(1,1)+Xi(2,1)*y(1)+Xi(3,1)*y(2)+Xi(4,1)*y(1)*y(2)+Xi(5,1)*y(2)^2;
        Xi(1,2)+Xi(2,2)*y(1)+Xi(3,2)*y(2)+Xi(4,2)*y(1)*y(2)+Xi(5,2)*y(2)^2];