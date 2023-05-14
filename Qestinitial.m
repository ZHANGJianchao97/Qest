function meop=Qestinitial(K,d)
    dimH=d; %by default this is qubit case,
    numPo=K; %number of POVM elements is K

    povm=cell(1,numPo);
    meop=cell(1,numPo);
    sumpovm=zeros(dimH,dimH);
    for i=1:numPo
        povm{i}=comprandn(dimH,dimH);
        povm{i}=(povm{i}+povm{i}')/2;
        povm{i}=povm{i}'*povm{i};
        sumpovm=sumpovm+povm{i};
    end
    G=povmSum(povm,numPo,dimH);
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