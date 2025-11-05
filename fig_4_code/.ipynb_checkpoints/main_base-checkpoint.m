clear all
close all
clc

%% Define parameter values:
global mu k C opt_base J_phi N L W_test b eta lambda T T_test; 

% Instructor:
    mu  = 2;

% Student:
    g   = 1;
    N   = 75;  % # basis functions
    L   = 10;  % # number of dopamine signals
    D   = 2;   % Dim of the state
    k   = 5;   % instructor feedback control gain

    eta = 1;
    C   = normrnd(0, 1 , D, L);     % structure/architecture

% Base dynamics:

    opt_base = 0; 
    % 0: base dynamics just contains the leakage term; 
    % 1: base dynamics is a bi-stable attractor
    
    % Define parameters of dynamics in each case:
    if opt_base == 0
        lambda = 1; 
    elseif opt_base == 1
        b = 1;
    end

%% Plot the distribution of basis functions in state-space:

    % Create mesh in state-space:
    [X,X_dot] = meshgrid(-5:0.1:5,-10:0.1:10);

    % Evaluate basis functions:
        % Create randomized matrix, J_phi:
          J_phi = normrnd(0, g^2 , N, D); 

        % Evaluate basis functions at the gridpoints:
          phi_mesh = NaN(size(X,1), size(X,2), N);

            for ii = 1:size(X,1)
                for jj = 1:size(X,2)
                    phi_mesh(ii,jj,:) = tanh(J_phi*[X(ii,jj); X_dot(ii,jj)])./sqrt(N);
                end
            end

    % Plot in 3D:
    figure;
    xlabel('$x$', 'Interpreter', 'latex','FontSize', 20); 
    ylabel('$\dot{x}$', 'Interpreter', 'latex','FontSize', 20); 
    zlabel('$\phi_i(x, \dot{x})$', 'Interpreter', 'latex','FontSize', 20 );
    box on;
    set(gca,'TickLabelInterpreter','latex', 'FontSize', 20);
    for kk = 1:N
        surf(X, X_dot, phi_mesh(:,:,kk)); hold on;
    end
    colorbar;



%% Plot the (autonomous, u = 0) base dynamics (i.e., W = 0):

 if opt_base == 0       % base: leakage term
    % Flow field (quiver):
    [X1,X2] = meshgrid(-2:0.1:2,-4:0.1:4);
    fun1    = -X1;
    fun2    = - X2;
 elseif opt_base == 1   % base: bi-stable attractor base dynamics
    % Flow field (quiver):
    [X1,X2] = meshgrid(-sqrt(b)*2:0.1:sqrt(b)*2,-sqrt(b)*4:0.1:sqrt(b)*4);
    fun1 = -(1/8)*(X1 + X2).^3 + b*X2;
    fun2 = -(1/8)*(X1 + X2).^3 + b*X1;
else 
    display('=== error with the base dynamics option selection ===')
end

% Plot base flow field, colored according to magnitude:
% ##### PRODUCES FIGS. 4B.1A (opt_base = 0) and 4B.2A (opt_base = 1), "before learning", IN THE MANUSCRIPT ########
figure;
imagesc([X1(1,1) X1(1,end)], [X2(1,1) X2(end,1)], sqrt(fun1.^2 + fun2.^2)); hold on;
quiver(X1(1:5:end,1:5:end), X2(1:5:end,1:5:end), fun1(1:5:end,1:5:end), fun2(1:5:end,1:5:end),'Color', 'w','linewidth',3); hold on;
xlabel('$x_1$', 'Interpreter', 'latex','FontSize', 20); 
ylabel('$x_2$', 'Interpreter', 'latex','FontSize', 20);
set(gca,'TickLabelInterpreter','latex', 'FontSize', 20);
set(gca,'YDir','normal');
c1 = colorbar;
c1.TickLabelInterpreter = 'latex';

% Include fixed points, colored according to their stability:
if opt_base == 0
    plot(0, 0, 'o', 'MarkerSize',6, 'MarkerEdgeColor', 'g', 'MarkerFaceColor','g','MarkerSize',10); hold on;
elseif opt_base == 1
    plot(sqrt(b),sqrt(b), 'o', 'MarkerSize',6, 'MarkerEdgeColor', 'g', 'MarkerFaceColor','g','MarkerSize',10); hold on;
    plot(-sqrt(b),-sqrt(b), 'o', 'MarkerSize',6, 'MarkerEdgeColor', 'g', 'MarkerFaceColor','g','MarkerSize',10); hold on;
    plot(0, 0, 'o', 'MarkerSize',6, 'MarkerEdgeColor', 'r', 'MarkerFaceColor','r','MarkerSize',10); hold on;
end



%% Training of the RNN:

T       = 400; % learning time-horizon
ICs     = [2; 0; 2; 0; reshape(zeros(N,L), [N*L, 1])];
%states(end,:);


times = 0;
states = ICs';

% Training:
% Active Learning loop:
display('========= RNN TRAINING =========');
[times_aux, states_aux] = ode45(@instructor_student_dynamics_base, times(end):0.05: times(end)+T, ICs);

times       = [times; times_aux];
states      = [states; states_aux];

% Reshape the learnt weights into matrix form:
W_test =  reshape(states(end, 5:end), [N,L]);

% Plot the learning dynamics:
[X1,X2,gg, f1, f2] = plotLearning(times, states);

display('========= END OF RNN TRAINING =========');
pause;


%% Test the student --- after learning, without feedback from the instructor --- for different stimuli:
display('========= RNN TESTING =========');
T_test   = 200;
IC_test  = [1; 0; 1; 0];

% Test learned RNN:
[times_test, states_test] = ode45(@instructor_student_dynamics_base_test,0:0.05:T_test, IC_test);

% Plot testing dynamics/student performance:
plotTesting(times_test, states_test, X1, X2, gg, f1, f2);
display('========= END OF RNN TESTING =========');

save("data_instructive_RNN.mat");

