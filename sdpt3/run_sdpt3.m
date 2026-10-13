% run_sdpt3.m

% 1. Initialize SDPT3 paths
startup;

% 2. Read data exported from C++
data = jsondecode(fileread('sdpt3_data.json'));
D = length(data.b);

% 3. Define block structure for Gamma, Q, and G (3 semidefinite cones)
blk = cell(3,2);
blk{1,1} = 's'; blk{1,2} = size(data.C_Gamma, 1);
blk{2,1} = 's'; blk{2,2} = size(data.C_Q, 1);
blk{3,1} = 's'; blk{3,2} = size(data.C_G, 1);

% 4. Define C matrices (the constant offsets)
C = cell(3,1);
C{1} = data.C_Gamma;
C{2} = data.C_Q;
C{3} = data.C_G;

b = data.b;

% 5. Define constraint matrices (At)
At = cell(3,1);
s_Gamma = blk{1,2}*(blk{1,2}+1)/2;
s_Q = blk{2,2}*(blk{2,2}+1)/2;
s_G = blk{3,2}*(blk{3,2}+1)/2;

At{1} = sparse(s_Gamma, D);
At{2} = sparse(s_Q, D);
At{3} = sparse(s_G, D);

% Vectorize each step matrix using SDPT3's svec function
for i = 1:D
    At{1}(:, i) = svec(blk(1,:), squeeze(data.A_Gamma(i,:,:)));
    At{2}(:, i) = svec(blk(2,:), squeeze(data.A_Q(i,:,:)));
    At{3}(:, i) = svec(blk(3,:), squeeze(data.A_G(i,:,:)));
end

% 6. Run the solver with default parameters (HKM direction)
[obj, X, y, Z, info] = sqlp(blk, At, C, b);

% 7. Calculate final Ground State Energy
% Add the static offset Tr(K * Gamma_init) back to the dynamic optimization part
offset_energy = sum(sum(data.K_mat .* data.C_Gamma));

% obj(2) is the dual objective (b^T y). Our formulation is minimized at offset - (b^T y)
sdp_energy = 2.0 * (offset_energy - obj(2));

fprintf('\n--- Oplossing Gevonden ---\n');
fprintf('Optimale Grondtoestandsenergie (SDPT3): %.10f\n', sdp_energy);