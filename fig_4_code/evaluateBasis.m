function [phi_val] = evaluateBasis(x, x_dot)
% Given scalar values of x and x_dot, phi_val provides the values of all
% basis functions considered at that point

global J_phi N ;

phi_val = tanh(J_phi*[x; x_dot])./sqrt(N);
 
end