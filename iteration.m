function [meop,ob,FIinv,d,prob,varargout]=iteration(rho,drho,meop,FIinv,d,prob,varargin)
        
    alpha=[1e+6 1e+4 1e+2 1 0.1 0.01 1e-4 1e-6];
    lalpha=length(alpha);
    min=1e+10;
    if isempty(varargin)
        for i=1:lalpha
            [~,ob,~,~,~]=iteration_step(rho,drho,meop,FIinv,d,prob,alpha(i));
            if ob<min
                position=i;
                min=ob;
            end
        end
        step=alpha(position);
        [meop,ob,FIinv,d,prob]=iteration_step(rho,drho,meop,FIinv,d,prob,step);
    else %conjugate gradient case
        oldH=varargin{1};
        oldP=varargin{2};
        [meop,~,newH,newP]=cal_povm3(rho,drho,meop,FIinv,d,prob,0.1,oldH,oldP);
        varargout{1}=newH;
        varargout{2}=newP;
        [ob,FIinv,d,prob]=meop_ob(rho,drho,meop);
    end

    %[newmeop,newpovm,newH,newP]=cal_povm3(meop,numPo,numSt,dim,rho,rhop,FIinv,prob,d,alpha,oldH,oldP)
end

function [meop,ob,FIinv,d,prob]=iteration_step(rho,drho,meop,FIinv,d,prob,step)
%ITERATION Summary of this function goes here
%   Detailed explanation goes here
    
    %[newmeop,newpovm]=cal_povm2(meop,numPo,numSt,dim,rho,rhop,FIinv,prob,d,alpha);
    dim=length(rho); 
    numPo=length(meop);
    numSt=length(drho);
    alpha=step; %step size

    Rho=cell(1,numSt); %Rho{i}=rho^i=\sum_j J^{-1}_{ji}\rho_j
    W=eye(numSt);
    FIinv=sqrt(W)*FIinv*sqrt(W);  %using the weight
    for i=1:numSt
        Rho{i}=zeros(dim);
        for j=1:numSt
            Rho{i}=Rho{i}+FIinv(j,i)*drho{j};
        end
    end
    
    D=FIinv*d;

    l=zeros(numSt,numPo);
    for k=1:numPo
        l(:,k)=D(:,k)/prob(k);
    end

    Y=cell(1,numPo);
    for k=1:numPo
        Y{k}=zeros(dim,dim);
        for i=1:numSt
            Y{k}=Y{k}+2*Rho{i}*l(i,k)-rho*l(i,k)*l(i,k);
        end
    end
    
    Lam=zeros(dim,dim);
    for k=1:numPo
        Lam=Lam+Y{k}'*meop{k}'*meop{k}+meop{k}'*meop{k}*Y{k};
    end
    Lam=Lam/2;
    
    %calcualte new povm by A+alphaH
    H=cell(1,numPo);
    newmeop=cell(1,numPo);
    newpovm=cell(1,numPo);
    for k=1:numPo
        H{k}=meop{k}*(Y{k}-Lam);
        newmeop{k}=meop{k}+alpha*H{k};
        newpovm{k}=newmeop{k}'*newmeop{k};
    end
    
    % normalization regarding A_k
    G=povmSum(newpovm,numPo,dim);
    if det(G)<1e-10
        warning('The determinant of summation of povm less than e-10');
    end
    Gis=sqrtm(inv(G));
    
    for k=1:numPo
        newmeop{k}=newmeop{k}*Gis;
        newpovm{k}=newmeop{k}'*newmeop{k};
    end    

    meop=newmeop;
    [ob,FIinv,d,prob]=meop_ob(rho,drho,meop);
end

function [newmeop,newpovm,newH,newP]=cal_povm3(rho,rhop,meop,FIinv,d,prob,alpha,oldH,oldP)
% Conjugate gradient  [meop,ob,FIinv,d,prob]=iteration_conjugate_step(rho,drho,meop,FIinv,d,prob,alpha(i))

    dim=length(rho); % qubit case default
    numPo=length(meop);
    numSt=length(rhop);
    betamethod=1;
    Rho=cell(1,numSt); %Rho{i}=rho^i
    
    for i=1:numSt
        Rho{i}=zeros(dim);
        for j=1:numSt
            Rho{i}=Rho{i}+FIinv(j,i)*rhop{j};
        end
    end
    D=zeros(numSt,numPo);
    for i=1:numSt
        for k=1:numPo
            for j=1:numSt
        D(i,k)=D(i,k)+FIinv(j,i)*d(j,k);
            end
        end
    end
    l=zeros(numSt,numPo);
    for i=1:numSt
        for k=1:numPo
            l(i,k)=D(i,k)/prob(k);
        end
    end
    
    Y=cell(1,numPo);
    for k=1:numPo
        Y{k}=zeros(dim,dim);
        for i=1:numSt
            Y{k}=Y{k}+2*Rho{i}*l(i,k)-rho*l(i,k)*l(i,k);
        end
    end
    
    Lam=zeros(dim,dim);
    for k=1:numPo
        Lam=Lam+Y{k}'*meop{k}'*meop{k}+meop{k}'*meop{k}*Y{k};
    end
    Lam=Lam/2;
    
    % calcualte new povm by A+alpha p_k^n
    newH=cell(1,numPo);
    beta=zeros(1,numPo);
    newP=cell(1,numPo);
    newmeop=cell(1,numPo);
    newpovm=cell(1,numPo);
    for k=1:numPo
        
        newH{k}=meop{k}*(Y{k}-Lam);  %get H_k^n
        
        if betamethod==1
            denominator=trace(oldP{k}'*(-newH{k}+oldH{k}));  %Dai-Yuan beta
            numerator=trace(newH{k}'*newH{k});
        elseif betamethod==2
            denominator=trace(oldH{k}'*oldH{k});  %Fletcher-Rieves beta
            numerator=-trace(newH{k}'*newH{k});
        elseif betamethod==3
            denominator=trace(oldH{k}'*oldH{k});  %Polak-Ribbiere beta
            numerator=-trace((-newH{k}+oldH{k})'*(-newH{k}));
        end

       
        if denominator==0
            beta(k)=0;
        else
            beta(k)=numerator/denominator; %get beta
            if(abs(beta(k))<1e-15)
                warning('beta less than 1e-15.')
            end
        end
        %disp(beta(k));
        newP{k}=-newH{k}-beta(k)*oldP{k}; % get newP
        newmeop{k}=meop{k}-alpha*newP{k}; %get newA
        newpovm{k}=newmeop{k}'*newmeop{k}; %get new povm
    end
    
    % normalization regarding A_k
    G=povmSum(newpovm,numPo,dim);
    if det(G)<1e-10
        warning('The determinant of summation of povm less than e-10');
    end
    Gis=sqrtm(inv(G));
    
    for k=1:numPo
        newmeop{k}=newmeop{k}*Gis;
        newpovm{k}=newmeop{k}'*newmeop{k};
    end    
end  % Conjugate gradient


 
%% draft functions
function re=povmSum(povm,numPo,dim)
    re=zeros(dim,dim);
    for i=1:numPo
        re=re+povm{i};
    end
end

%   truncate the imaginary part of diagonal entries.
function outpovm=realdiag(povm)
    outpovm=povm-diag(imag(diag(povm))*1j);
end
