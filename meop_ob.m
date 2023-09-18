function [ob,FIinv,d,prob] = meop_ob(rho,drho,meop,W)
% get the current objective function
% get the current FIinv, d, prob to prepare for next stage
    numPo=length(meop);
    numSt=length(drho);
    povm=cell(1,numPo);
    for k=1:numPo
        povm{k}=meop{k}'*meop{k};
    end  
    prob=zeros(1,numPo);
    for k=1:numPo
        prob(k)=trace(povm{k}*rho);
        % the real truncation
        if imag(prob(k))>1e-5
            warning('real truncation delete imaginary part larger than 1e-5');
            disp('imag(prob(k))');
            disp(imag(prob(k)));
        end
        prob(k)=real(prob(k)); 
    end
    d=zeros(numSt,numPo);
    for i=1:numSt
        for k=1:numPo
            d(i,k)=trace(povm{k}*drho{i});
            % the real truncation
            if imag(d(i,k))>1e-5
            warning('real truncation delete imaginary part larger than 1e-5');
            disp('imag(d(i,k))');
            disp(imag(d(i,k)));
            end
            d(i,k)=real(d(i,k)); 
        end
    end
    FI=zeros(numSt,numSt);
    for i=1:numSt
        for j=1:numSt
            for k=1:numPo
                FI(i,j)=FI(i,j)+d(i,k)*d(j,k)/prob(k);
            end
        end
    end
    rankFI=rank(FI);
    if rankFI<numSt
        disp(FI);
        error('Information matrix is Singular.')
    end
    FIinv=inv(FI);
    ob=trace(FI\W);
end





