function [meop,rho,drho]=Qestinitial_tensor(rho,drho,K,d,M)
    dimH=d; %by default this is qubit case,
    numPo=K; %number of POVM elements is K
    numSt=length(drho);
    I=eye(dimH);
    %rho=[epslon  0;
    %    0 1-epslon];
    rhoM=getrhoM(rho,M);

    drhoM=cell(1,numSt);
    for i=1:numSt
        %drhoi=drho{i};
        drhoM{i}=tensor(drho{i},rho)+tensor(rho,drho{i});
        for j=3:M
            drhoM{i}=tensor(drhoM{i},rho)+tensor(getrhoM(rho,j-1),drho{i});
        end
    end
    drho=drhoM;
    rho=rhoM;


    povm=cell(1,numPo);
    meop=cell(1,numPo);
    pdim=dimH^M;  %pdim is the dimension of the povm matrix
    sumpovm=zeros(pdim,pdim);
    for i=1:numPo
        povm{i}=comprandn(pdim,pdim);
        povm{i}=(povm{i}+povm{i}')/2;
        povm{i}=povm{i}'*povm{i};
        sumpovm=sumpovm+povm{i};
    end
    G=povmSum(povm,numPo,pdim);
    if det(G)<1e-10
        warning('The determinant of summation of povm less than e-10');
    end
    Gis=sqrtm(inv(G));
    for k=1:numPo
        povm{k}=Gis*povm{k}*Gis;
        meop{k}=chol(realdiag(povm{k}));  %get A_k
    end
end



function output=comprandn(r,c) %get random complex r*c matrix
re=randn(r,c);
im=randn(r,c)*1j;
output=re+im;
end

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

function rhoM=getrhoM(rho,M)
    rhoM=rho;
    for i=2:M
        rhoM=tensor(rhoM,rho);
    end
end

function re = tensor(A,B)
%TENSOR Summary of this function goes here
%   Detailed explanation goes here
    [a, b]=size(A);
    [m, n]=size(B);
    re=zeros(a*m,b*n);
    for i=1:a
        for j=1:b
            re(i*m-m+1:i*m,j*n-n+1:j*n)=A(i,j)*B;
        end
    end
end

