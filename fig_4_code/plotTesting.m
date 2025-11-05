 function [ ] = plotTesting(times_test, states_test, X1, X2, gg, f1, f2)
% Plot performance of the student when forced by different stimuli after
% learning -- no feedback from the instructor.

global mu W_test C;

%% Plot teacher and student trajectories:
% in phase space:
figure; hold on;
plot(states_test(:,1),states_test(:,2), 'm'); hold on;
plot(states_test(:,3),states_test(:,4), 'r'); hold on;
box on;
xlabel('$z$', 'Interpreter', 'latex','FontSize', 20); 
ylabel('$\dot{z}$', 'Interpreter', 'latex','FontSize', 20); 
set(gca,'TickLabelInterpreter','latex', 'FontSize', 20);
title('Instructor and student (testing)')

% in time:
figure; hold on;
plot(times_test, states_test(:,1), 'm'); hold on;
plot(times_test, states_test(:,3), 'r'); hold on;
box on;
xlabel('$t$', 'Interpreter', 'latex','FontSize', 20); 
ylabel('$z,x_1$', 'Interpreter', 'latex','FontSize', 20); 
set(gca,'TickLabelInterpreter','latex', 'FontSize', 20);
title('Instructor and student (testing)');

figure; hold on;
plot(times_test, states_test(:,2), 'm'); hold on;
plot(times_test, states_test(:,4), 'r'); hold on;
box on;
xlabel('$t$', 'Interpreter', 'latex','FontSize', 20); 
ylabel('$\dot{z},x_2$', 'Interpreter', 'latex','FontSize', 20); 
set(gca,'TickLabelInterpreter','latex', 'FontSize', 20);
title('Instructor and student (testing)')

% in phase space, with quiver:
% ###### GENERATES FIG 4B.1B (opt_base == 0) or FIG 4B.2B (opt_base == 1), "after learning", OF THE MANUSCRIPT
figure;
imagesc([X1(1,1) X1(1,end)], [X2(1,1) X2(end,1)], sqrt(gg(1:end/2,:).^2 + gg(end/2+1:end,:).^2)); hold on;
set(gca,'YDir','normal');
plot(states_test(round(end/2):end,3),states_test(round(end/2):end,4), 'w', 'lineWidth', 2); hold on;
 plot(0,0,'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r', 'MarkerSize', 5); hold on;
quiver(X1(1:5:end,1:5:end), X2(1:5:end,1:5:end), gg(1:5:end/2,1:5:end), gg(end/2+1:5:end,1:5:end),4, 'Color', 'w', 'lineWidth', 2); hold on;
c = colorbar;
xlabel('$x_1$', 'Interpreter', 'latex','FontSize', 20); 
ylabel('$x_2$', 'Interpreter', 'latex','FontSize', 20);
set(gca,'TickLabelInterpreter','latex', 'FontSize', 20);
c.Label.Interpreter = 'latex';
xlim([-2 2]); ylim([-4 4]);
box on;

 % ###### GENERATES "VAN DER POL" PANEL IN FIG.4B OF THE MANUSCRIPT
 % (not testing the RNN, just simulating the true dynamics)
 figure; 
 imagesc([X1(1,1) X1(1,end)], [X2(end,1) X2(1,1)] , sqrt(f1.^2 + f2.^2)); hold on;
 set(gca,'YDir','normal');
 quiver(X1(1:5:end,1:5:end), X2(1:5:end,1:5:end), f1(1:5:end,1:5:end), f2(1:5:end,1:5:end), 4, 'Color', 'w', 'lineWidth', 2); hold on;
 plot(states_test(round(end/2):end,1), states_test(round(end/2):end,2), 'w', 'lineWidth', 2); hold on;
 plot(0,0,'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r', 'MarkerSize', 5); hold on;
 xlabel('$z$', 'Interpreter', 'latex','FontSize', 20); 
 ylabel('$\dot{z}$', 'Interpreter', 'latex','FontSize', 20); 
 set(gca,'TickLabelInterpreter','latex', 'FontSize', 20);
 xlim([-2 2]); ylim([-4 4]);
 c = colorbar;
 c.Label.Interpreter = 'latex';

end