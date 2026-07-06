#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Compute gravitational and gas drag accelerations for all active bodies
void compute_accelerations(const double *pos, const double *vel, const double *gms, int num_bodies, 
                           double gas_eta, double fill_val_eta, double tau_base, double drag_beta, double gm_ref, double *acc) {
    // Note: We use gas_eta parameter for the gas headwind factor η.
    
    // Reset accelerations
    for (int i = 0; i < num_bodies * 3; i++) {
        acc[i] = 0.0;
    }

    // 1. Gravitational forces (pairwise interaction)
    for (int i = 0; i < num_bodies; i++) {
        if (gms[i] <= 0.0) continue; // Skip inactive bodies

        for (int j = 0; j < num_bodies; j++) {
            if (i == j || gms[j] <= 0.0) continue;

            double dx = pos[j * 3 + 0] - pos[i * 3 + 0];
            double dy = pos[j * 3 + 1] - pos[i * 3 + 1];
            double dz = pos[j * 3 + 2] - pos[i * 3 + 2];
            double dist2 = dx * dx + dy * dy + dz * dz;
            double dist = sqrt(dist2);

            // Numerical softening to prevent singularities
            if (dist < 1e-4) {
                dist = 1e-4;
                dist2 = 1e-8;
            }

            double factor = gms[j] / (dist2 * dist);
            acc[i * 3 + 0] += factor * dx;
            acc[i * 3 + 1] += factor * dy;
            acc[i * 3 + 2] += factor * dz;
        }
    }

    // 2. Gas Drag forces (only for planetesimals, i.e., i > 0, relative to the star at index 0)
    if (gms[0] > 0.0) {
        double star_x = pos[0 * 3 + 0];
        double star_y = pos[0 * 3 + 1];
        double star_z = pos[0 * 3 + 2];

        for (int i = 1; i < num_bodies; i++) {
            if (gms[i] <= 0.0) continue;

            // Vector relative to central star
            double rx = pos[i * 3 + 0] - star_x;
            double ry = pos[i * 3 + 1] - star_y;
            double rz = pos[i * 3 + 2] - star_z;
            double r2 = rx * rx + ry * ry + rz * rz;
            double r = sqrt(r2);

            if (r < 1e-3) continue; // Inside or extremely close to the star

            // Keplerian velocity magnitude v_K = sqrt(GM_star / r)
            double v_K = sqrt(gms[0] / r);

            // Projection onto xy plane for azimuthal direction (assuming rotation counter-clockwise)
            double r_xy = sqrt(rx * rx + ry * ry);
            if (r_xy < 1e-4) continue;

            // Gas velocity vector: circular rotation at (1 - gas_eta) * v_K
            double v_gas_x = -(1.0 - gas_eta) * v_K * (ry / r_xy);
            double v_gas_y = (1.0 - gas_eta) * v_K * (rx / r_xy);
            double v_gas_z = 0.0;

            // Relative velocity between planetesimal and gas headwind
            double v_rel_x = vel[i * 3 + 0] - v_gas_x;
            double v_rel_y = vel[i * 3 + 1] - v_gas_y;
            double v_rel_z = vel[i * 3 + 2] - v_gas_z;

            // Drag timescale scales with body mass: tau = tau_base * (m / m_ref)^beta
            double mass_ratio = gms[i] / gm_ref;
            if (mass_ratio < 1e-10) mass_ratio = 1e-10;
            
            double tau = tau_base * pow(mass_ratio, drag_beta);
            if (tau < 1e-2) tau = 1e-2;

            // Apply drag acceleration
            acc[i * 3 + 0] -= v_rel_x / tau;
            acc[i * 3 + 1] -= v_rel_y / tau;
            acc[i * 3 + 2] -= v_rel_z / tau;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 13) {
        printf("Usage: %s <num_bodies> <years> <dt> <save_every> <r_scale> <gas_eta> <tau_base> <drag_beta> <gm_ref> <input_bin> <output_bin> <collisions_log>\n", argv[0]);
        return 1;
    }

    int num_bodies = atoi(argv[1]);
    double years = atof(argv[2]);
    double dt = atof(argv[3]);
    int save_every = atoi(argv[4]);
    double r_scale = atof(argv[5]);
    double gas_eta = atof(argv[6]);
    double fill_val_eta = gas_eta; // kept for compatibility if needed
    double tau_base = atof(argv[7]);
    double drag_beta = atof(argv[8]);
    double gm_ref = atof(argv[9]);
    const char *input_path = argv[10];
    const char *output_path = argv[11];
    const char *collisions_path = argv[12];

    double total_days = years * 365.25;
    long long num_steps = (long long)ceil(total_days / dt);

    // Allocate memory
    double *gms = malloc(num_bodies * sizeof(double));
    double *pos = malloc(num_bodies * 3 * sizeof(double));
    double *vel = malloc(num_bodies * 3 * sizeof(double));
    double *acc = malloc(num_bodies * 3 * sizeof(double));
    double *vel_half = malloc(num_bodies * 3 * sizeof(double));

    if (!gms || !pos || !vel || !acc || !vel_half) {
        printf("Error: Failed to allocate memory.\n");
        return 1;
    }

    // Read input binary file
    FILE *fin = fopen(input_path, "rb");
    if (!fin) {
        perror("Error: Failed to open input file");
        free(gms); free(pos); free(vel); free(acc); free(vel_half);
        return 1;
    }
    fread(gms, sizeof(double), num_bodies, fin);
    fread(pos, sizeof(double), num_bodies * 3, fin);
    fread(vel, sizeof(double), num_bodies * 3, fin);
    fclose(fin);

    // Open output and collision files
    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        perror("Error: Failed to open output file");
        free(gms); free(pos); free(vel); free(acc); free(vel_half);
        return 1;
    }

    FILE *fcol = fopen(collisions_path, "w");
    if (!fcol) {
        perror("Error: Failed to open collisions log file");
        fclose(fout);
        free(gms); free(pos); free(vel); free(acc); free(vel_half);
        return 1;
    }

    // Write CSV header for collision log
    fprintf(fcol, "time_years,survivor_idx,merged_idx,survivor_mass_before_earth,merged_mass_before_earth,new_mass_earth,dist_au\n");

    // Compute initial accelerations
    compute_accelerations(pos, vel, gms, num_bodies, gas_eta, fill_val_eta, tau_base, drag_beta, gm_ref, acc);

    // Write initial step (step 0)
    double t_zero = 0.0;
    fwrite(&t_zero, sizeof(double), 1, fout);
    fwrite(gms, sizeof(double), num_bodies, fout);
    fwrite(pos, sizeof(double), num_bodies * 3, fout);
    fwrite(vel, sizeof(double), num_bodies * 3, fout);

    // Main simulation loop (Velocity Verlet)
    for (long long step = 1; step <= num_steps; step++) {
        double t_curr = step * dt;

        // 1. Position update: x(t+dt) = x(t) + v(t+0.5dt) * dt
        //    And Velocity half-step update: v(t+0.5dt) = v(t) + 0.5 * a(t) * dt
        for (int i = 0; i < num_bodies * 3; i++) {
            int body_idx = i / 3;
            if (gms[body_idx] <= 0.0) continue; // Skip inactive bodies

            vel_half[i] = vel[i] + 0.5 * acc[i] * dt;
            pos[i] = pos[i] + vel_half[i] * dt;
        }

        // 2. Recompute accelerations: a(t+dt) based on new positions and velocity half-steps
        compute_accelerations(pos, vel_half, gms, num_bodies, gas_eta, fill_val_eta, tau_base, drag_beta, gm_ref, acc);

        // 3. Velocity full-step update: v(t+dt) = v(t+0.5dt) + 0.5 * a(t+dt) * dt
        for (int i = 0; i < num_bodies * 3; i++) {
            int body_idx = i / 3;
            if (gms[body_idx] <= 0.0) continue;

            vel[i] = vel_half[i] + 0.5 * acc[i] * dt;
        }

        // 4. Collision checking and coalescence merging
        for (int i = 0; i < num_bodies; i++) {
            if (gms[i] <= 0.0) continue;

            // Star (index 0) has a fixed accretion radius of 0.05 AU (approx 10 solar radii)
            // Other planetesimals scale with their mass
            double ri = (i == 0) ? 0.05 : r_scale * pow(gms[i], 1.0 / 3.0);

            for (int j = i + 1; j < num_bodies; j++) {
                if (gms[j] <= 0.0) continue;

                double rj = r_scale * pow(gms[j], 1.0 / 3.0); // Body j is always a planetesimal (j > i >= 0)

                double dx = pos[j * 3 + 0] - pos[i * 3 + 0];
                double dy = pos[j * 3 + 1] - pos[i * 3 + 1];
                double dz = pos[j * 3 + 2] - pos[i * 3 + 2];
                double dist = sqrt(dx*dx + dy*dy + dz*dz);

                // If distance is less than the sum of collision radii, they merge
                if (dist < (ri + rj)) {
                    double m_i_before = gms[i];
                    double m_j_before = gms[j];
                    double m_new = m_i_before + m_j_before;

                    // Center of mass position (conserves center of mass coordinate)
                    double new_x = (m_i_before * pos[i * 3 + 0] + m_j_before * pos[j * 3 + 0]) / m_new;
                    double new_y = (m_i_before * pos[i * 3 + 1] + m_j_before * pos[j * 3 + 1]) / m_new;
                    double new_z = (m_i_before * pos[i * 3 + 2] + m_j_before * pos[j * 3 + 2]) / m_new;

                    // Center of mass velocity (conserves momentum)
                    double new_vx = (m_i_before * vel[i * 3 + 0] + m_j_before * vel[j * 3 + 0]) / m_new;
                    double new_vy = (m_i_before * vel[i * 3 + 1] + m_j_before * vel[j * 3 + 1]) / m_new;
                    double new_vz = (m_i_before * vel[i * 3 + 2] + m_j_before * vel[j * 3 + 2]) / m_new;

                    // Decide survivor based on larger mass (physical logic)
                    int survivor = i;
                    int merged = j;
                    if (m_j_before > m_i_before) {
                        survivor = j;
                        merged = i;
                    }

                    gms[survivor] = m_new;
                    pos[survivor * 3 + 0] = new_x;
                    pos[survivor * 3 + 1] = new_y;
                    pos[survivor * 3 + 2] = new_z;
                    vel[survivor * 3 + 0] = new_vx;
                    vel[survivor * 3 + 1] = new_vy;
                    vel[survivor * 3 + 2] = new_vz;

                    // Merged body becomes inactive
                    gms[merged] = 0.0;
                    pos[merged * 3 + 0] = 0.0;
                    pos[merged * 3 + 1] = 0.0;
                    pos[merged * 3 + 2] = 0.0;
                    vel[merged * 3 + 0] = 0.0;
                    vel[merged * 3 + 1] = 0.0;
                    vel[merged * 3 + 2] = 0.0;

                    // Log the collision (convert time to years and masses to Earth masses using gm_ref)
                    double m_surv_before = (survivor == i) ? m_i_before : m_j_before;
                    double m_merg_before = (survivor == i) ? m_j_before : m_i_before;
                    fprintf(fcol, "%f,%d,%d,%f,%f,%f,%f\n", t_curr / 365.25, survivor, merged, 
                            m_surv_before / gm_ref, m_merg_before / gm_ref, m_new / gm_ref, dist);
                    fflush(fcol);

                    // If body i was the one merged/destroyed, we must stop checking it in this step
                    if (survivor == j) {
                        break;
                    }

                    // Recalculate radius of survivor for subsequent checks in this loop
                    ri = (i == 0) ? 0.05 : r_scale * pow(gms[i], 1.0 / 3.0);
                }
            }
        }

        // 5. Output state periodically
        if (step % save_every == 0 || step == num_steps) {
            fwrite(&t_curr, sizeof(double), 1, fout);
            fwrite(gms, sizeof(double), num_bodies, fout);
            fwrite(pos, sizeof(double), num_bodies * 3, fout);
            fwrite(vel, sizeof(double), num_bodies * 3, fout);
        }
    }

    // Clean up
    fclose(fout);
    fclose(fcol);
    free(gms);
    free(pos);
    free(vel);
    free(acc);
    free(vel_half);

    return 0;
}
