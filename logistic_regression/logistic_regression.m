function logistic_regression( a, b, c)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
A=importdata(a);
deg=str2num(b);
[il,ib]=size(A);
for i=1:il
    if A(i,ib)~=1
        A(i,ib)=0;
    end
end
weights=zeros((deg*(ib-1))+1,1);
totalweight=1;
ent=1;
while totalweight>=0.001 && ent>=0.001
    y=zeros(il,1);
    phi=zeros(il,deg*(ib-1)+1);
    t=A(:,ib);
    if deg==1
        for i=1:il
            func=zeros(ib,1);
            func(1,1)=1;
            for j=1:ib-1
                func(j+1,1)=A(i,j);
            end
            phi(i,:)=func;
            mid=transpose(weights)*func;
            y(i,1)=1/(1+exp(-(mid)));
        end
    end
    
    if deg==2
        for i=1:il
            func=zeros((ib-1)*2+1,1);
            func(1,1)=1;
            count=2;
            for j=1:ib-1
                func(count,1)=A(i,j);
                count=count+1;
                func(count,1)=A(i,j)*A(i,j);
                count=count+1;
            end
            phi(i,:)=func;
            mid=transpose(weights)*func;
            y(i,1)=1/(1+exp(-(mid)));
        end
       
    end
    
    R=zeros(il);
    for i=1:il
            R(i,i)=y(i,1)*(1-y(i,1));
    end
    temp=inv(transpose(phi)*R*phi)*(transpose(phi)*(y-t));
    new_weights=weights-temp;
    totalweight=abs(sum(new_weights-weights));
    weights=new_weights;
    new_ent=0.00;
    for i=1:il
        new_ent=new_ent+((t(i,1)*log(y(i,1)))+((1-t(i,1))*log((1-y(i,1)))));
    end
    new_ent=-new_ent;
    ent=abs(ent-new_ent);
   
    
end
disp(size(phi));
for i=1:deg*(ib-1)+1
    fprintf('w%d=%.4f\n',i-1,weights(i,1))
end

test=importdata(c);
[m,n]=size(test);


for i=1:m
    if test(i,n)~=1
        test(i,n)=0;
    end
end
totalaccuracy=0;
for i=1:m
    if deg==1
        func=zeros(n,1);
        func(1,1)=1;
        for j=1:ib-1
                func(j+1,1)=test(i,j);
         end
    end
    if deg==2
        func=zeros(deg*(n-1)+1,1);
        func(1,1)=1;
        count=2;
            for j=1:n-1
                func(count,1)=test(i,j);
                count=count+1;
                func(count,1)=test(i,j)*test(i,j);
                count=count+1;
            end
    end
    inner=transpose(weights)*func;
    y=1/(1+exp(-(inner)));
    if y>0.5
        predicted_class=1;
        tie=false;
    end
    if y<0.5
        predicted_class=0;
        y=1-y;
        tie=false;
    end
if y==0.5
        r=rand;
        tie=true;
        if r<=0.5
            predicted_class=1;
        else
            predicted_class=0;
        end
end
    
    if predicted_class==test(i,n) && tie==false
        accuracy=1;
    end
    if predicted_class~=test(i,n) && tie==false
        accuracy=0;
    end
    if tie==true
        accuracy=0.5;
    end
    totalaccuracy=totalaccuracy+accuracy;
    fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',i-1,predicted_class,y,test(i,n),accuracy);
    
end
fin=totalaccuracy/m;
fprintf('classification accuracy=%6.4f\n',fin);
    
end



