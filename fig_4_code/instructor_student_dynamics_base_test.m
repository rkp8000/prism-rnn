function [dstate_dt] = instructor_student_dynamics_base_test(t, state)
% Computes  derivative of the instructor-student dynamics for the
% Van der Pol instructor after learning (so they evolve independently)
% student dynamics includes a bi-stable base. 

global mu k C opt_basis J_phi N L W_test b opt_base lambda T_test; 

message         = [num2str(t), ' out of ', num2str(T_test) ];
display(num2str(message));
dstate_dt = NaN(size(state));
dstate_dt = NaN(size(state));

%% STIMULI SELECTION:

%stimuli = normrnd(0, 1);
% stimuli = sin(3*t);
stimuli = 0;

%% DYNAMICS:

phi   = evaluateBasis(state(3,1), state(4,1));

% Instructor dynamics (Van der Pol):
    dzdt = [state(2,1); -mu*(state(1,1)^2 - 1)*state(2,1) - state(1,1)] - [0; stimuli];

% Student dynamics (independent from instructor):
if opt_base == 0
    dxdt = -lambda*state(3:4,1) + C*W_test'*phi - [0; stimuli];
elseif opt_base == 1
    dxdt = [-(1/8)*(state(3,1) + state(4,1))^3 + b*state(4,1); ...
            -(1/8)*(state(3,1) + state(4,1))^3 + b*state(3,1)] + C*W_test'*phi - [0; stimuli];
end

% Return derivatives:
dstate_dt(1:4,1)   = [dzdt; dxdt];





end