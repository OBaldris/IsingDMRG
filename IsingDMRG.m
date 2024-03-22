%%DMRG of the Ising model code
clear all

chi = 16; 
Nsites =100;

sweeps = 3; 
updateon = 1;
OPTS.maxit = 2; 
OPTS.krydim = 4; 

chid = 2;
X = [0,1;1,0]; Y = [0,-1i;1i,0]; 
Z = [1,0;0,-1]; I = eye(2);
M = zeros(3,3,2,2);
M(1,1,:,:) = I; 
M(1,2,:,:) = -X;
M(2,3,:,:) = X;
M(1,3,:,:) = -Z; 
M(3,3,:,:) = I;
ML = reshape([1;0;0],[3,1,1]); 
MR = reshape([0;0;1],[3,1,1]); 

A = {};
A{1} = rand(1,chid,min(chi,chid));

A = {};
A{1} = rand(1,chid,min(chi,chid));
for k = 2:Nsites
    A{k} = rand(size(A{k-1},3),chid,min(min(chi,size(A{k-1},3)*chid),chid^(Nsites-k)));
end

Nsites = length(A);
L{1} = ML; 
R{Nsites} = MR;
chid = size(M,3); 
for p = 1:Nsites - 1
    chil = size(A{p},1); 
    chir = size(A{p},3);
    [qtemp,rtemp] = qr(reshape(A{p},[chil*chid,chir]),0);
    A{p} = reshape(qtemp,[chil,chid,chir]);
    A{p+1} = ncon({rtemp,A{p+1}},{[-1,1],[1,-2,-3]})/norm(rtemp(:));
    L{p+1} = ncon({L{p},M,A{p},conj(A{p})},{[2,1,4],[2,-1,3,5],[4,5,-3],[1,3,-2]});
end

chil = size(A{Nsites},1); chir = size(A{Nsites},3);
[qtemp,stemp] = qr(reshape(A{Nsites},[chil*chid,chir]),0);
A{Nsites} = reshape(qtemp,[chil,chid,chir]);
sWeight{Nsites+1} = stemp./sqrt(trace(stemp*stemp'));

Ekeep = [];
for k = 1:sweeps+1
    if k == sweeps+1
        updateon = 0;
    end
    
  
    for p = Nsites-1:-1:1
        chil = size(A{p},1); chir = size(A{p+1},3);
        psiGround = reshape(ncon({A{p},A{p+1},sWeight{p+2}},{[-1,-2,1],[1,-3,2],[2,-4]}),[chil*chid^2*chir,1]);
        
        if updateon 
            [psiGround,Ekeep(end+1)] = eigLanczos(psiGround,OPTS,@doApplyMPO,{L{p},M,M,R{p+1}});
        end
        
        [utemp,stemp,vtemp] = svd(reshape(psiGround,[chil*chid,chid*chir]),'econ');
        chitemp = min(min(size(stemp)),chi);
        A{p} = reshape(utemp(:,1:chitemp),[chil,chid,chitemp]);
        sWeight{p+1} = stemp(1:chitemp,1:chitemp)./sqrt(sum(diag(stemp(1:chitemp,1:chitemp)).^2));
        B{p+1} = reshape(vtemp(:,1:chitemp)',[chitemp,chid,chir]);
            
        R{p} = ncon({M,R{p+1},B{p+1},conj(B{p+1})},{[-1,2,3,5],[2,1,4],[-3,5,4],[-2,3,1]});
    end
    
    chil = size(A{1},1); chir = size(A{1},3);
    [utemp,stemp,vtemp] = svd(reshape(ncon({A{1},sWeight{2}},{[-1,-2,1],[1,-3]}),[chil,chid*chir]),'econ');
    B{1} = reshape(vtemp',[chil,chid,chir]);
    sWeight{1} = utemp*stemp./sqrt(trace(stemp.^2));
    
    for p = 1:Nsites-1
        chil = size(B{p},1); chir = size(B{p+1},3);
        psiGround = reshape(ncon({sWeight{p},B{p},B{p+1}},{[-1,1],[1,-2,2],[2,-3,-4]}),[chil*chid^2*chir,1]);
        if updateon 
            [psiGround,Ekeep(end+1)] = eigLanczos(psiGround,OPTS,@doApplyMPO,{L{p},M,M,R{p+1}});
        end
        [utemp,stemp,vtemp] = svd(reshape(psiGround,[chil*chid,chid*chir]),'econ');
        chitemp = min(min(size(stemp)),chi);
        A{p} = reshape(utemp(:,1:chitemp),[chil,chid,chitemp]);
        sWeight{p+1} = stemp(1:chitemp,1:chitemp)./sqrt(sum(diag(stemp(1:chitemp,1:chitemp)).^2));
        B{p+1} = reshape(vtemp(:,1:chitemp)',[chitemp,chid,chir]);
            
       
        L{p+1} = ncon({L{p},M,A{p},conj(A{p})},{[2,1,4],[2,-1,3,5],[4,5,-3],[1,3,-2]});
        
      
        
        fprintf('Sweep: %2.1d of %2.1d, Loc: %2.1d, Energy: %12.12d\n',k,sweeps,p,Ekeep(end));
        
    end
    
    
    chil = size(B{Nsites},1); chir = size(B{Nsites},3);
    [utemp,stemp,vtemp] = svd(reshape(ncon({B{Nsites},sWeight{Nsites}},{[1,-2,-3],[-1,1]}),[chil*chid,chir,1]),'econ');    
    A{Nsites} = reshape(utemp,[chil,chid,chir]);
    sWeight{Nsites+1} = (stemp./sqrt(sum(diag(stemp).^2)))*vtemp';
    
end
A{Nsites} = ncon({A{Nsites},sWeight{Nsites+1}},{[-1,-2,1],[1,-3]});
sWeight{Nsites+1} = eye(size(A{Nsites},3));




function psi = doApplyMPO(psi,L,M1,M2,R)
psi = reshape(ncon({reshape(psi,[size(L,3),size(M1,4),size(M2,4),size(R,3)]),L,M1,M2,R},...
    {[1,3,5,7],[2,-1,1],[2,4,-2,3],[4,6,-3,5],[6,-4,7]}),[size(L,3)*size(M1,4)*size(M2,4)*size(R,3),1]);
end


function [psivec,dval] = eigLanczos(psivec,OPTS,linFunct,functArgs)

if norm(psivec) == 0
    psivec = rand(length(psivec),1);
end
psi = zeros(numel(psivec),OPTS.krydim+1);
A = zeros(OPTS.krydim,OPTS.krydim);
for k = 1:OPTS.maxit
    
    psi(:,1) = psivec(:)/norm(psivec);
    for p = 2:OPTS.krydim+1
        psi(:,p) = linFunct(psi(:,p-1),functArgs{(1:length(functArgs))});
        for g = 1:1:p-1
            A(p-1,g) = dot(psi(:,p),psi(:,g));
            A(g,p-1) = conj(A(p-1,g));
        end
        for g = 1:1:p-1
            psi(:,p) = psi(:,p) - dot(psi(:,g),psi(:,p))*psi(:,g);
            psi(:,p) = psi(:,p)/max(norm(psi(:,p)),1e-16);
        end
    end
    
    [utemp,dtemp] = eig(0.5*(A+A'));
    xloc = find(diag(dtemp) == min(diag(dtemp)));
    psivec = psi(:,1:OPTS.krydim)*utemp(:,xloc(1));
end

psivec = psivec/norm(psivec);
dval = dtemp(xloc(1),xloc(1));
end









