function [dstate_dt] = instructor_student_dynamics_base(t, state)
% Instructor-teacher dynamics with bi-stable "base" dynamics in the
% student. Instructor is a Van der Pol oscillator. 

global mu k C opt_basis opt_base J_phi N L b eta lambda T; 

message         = [num2str(t), ' out of ', num2str(T) ];
display(num2str(message));
dstate_dt = NaN(size(state));

%% RESHAPE W FROM VECTORIZATION:

W       = reshape(state(5:end,1), [N,L]);

%% STIMULI SELECTION:

stimuli = normrnd(0, 2);
% stimuli = 2*sin(t);


%% DYNAMICS:

error = (state(1:2,1) - state(3:4,1));
phi   = evaluateBasis(state(3,1), state(4,1));

% Instructor dynamics (Van der Pol):
    dzdt = [state(2,1); -mu*(state(1,1)^2 - 1)*state(2,1) - state(1,1)] - [0; stimuli];

% Student (learning) dynamics:
%if opt_base == 0
    dxdt = -lambda*state(3:4,1) + C*W'*phi - [0; stimuli] + k*error;

%elseif opt_base == 1
  %  dxdt = [-(1/8)*(state(3,1) + state(4,1))^3 + b*state(4,1); ...
   %         -(1/8)*(state(3,1) + state(4,1))^3 + b*state(3,1)] + C*W'*phi - [0; stimuli] + k*error;
%end

% Weight dynamics (plasticity):
    dWdt = eta*phi*error'*C;

%% VECTORIZE & RETURN:

dstate_dt(1:4,1)   = [dzdt; dxdt];
dstate_dt(5:end,1) = dWdt(:);


end