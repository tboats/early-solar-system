import numpy as np
import os
import argparse
import subprocess
import math

def run_simulation(preset="solar_system", years=800.0, dt=0.2):
    src_dir = os.path.dirname(__file__)
    data_dir = os.path.join(src_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # ----------------------------------------------------
    # Constants & Conversion factors (JPL DE440 based)
    # ----------------------------------------------------
    AU_IN_KM = 149597870.7
    DAY_IN_S = 86400.0
    GM_CONVERSION = (DAY_IN_S ** 2) / (AU_IN_KM ** 3) # ~2.22749449e-15
    
    GM_SUN_KM3_S2 = 1.32712440041279419e11
    GM_SUN = GM_SUN_KM3_S2 * GM_CONVERSION # ~0.000295912208 AU^3/day^2
    
    # Earth Mass in Solar Masses
    M_EARTH_SOLAR = 3.003e-6
    GM_EARTH = M_EARTH_SOLAR * GM_SUN

    # Preset settings
    if preset == "solar_system":
        num_bodies = 240
        total_disk_mass_earth = 30.0
        gas_eta = 0.002       # Weak pressure support (weak headwind)
        tau_base = 25000.0    # Long drag timescale (weak drag in days)
        drag_beta = 0.33      # Drag timescale scales as mass^(1/3)
        r_scale = 30.0        # Collision radius inflation factor
        r_min, r_max = 0.5, 10.0
    elif preset == "hot_jupiter":
        num_bodies = 300
        total_disk_mass_earth = 80.0
        gas_eta = 0.008       # Strong pressure support (strong headwind)
        tau_base = 4500.0     # Short drag timescale (strong drag in days)
        drag_beta = 0.33      # Drag timescale scales as mass^(1/3)
        r_scale = 40.0        # Collision radius inflation factor
        r_min, r_max = 0.5, 12.0
    else:
        raise ValueError(f"Unknown preset: {preset}")

    # Set duration and stepping
    total_days = years * 365.25
    num_steps = int(np.ceil(total_days / dt))
    
    # Keep output file reasonable size
    max_saved_states = 6000
    save_every = max(1, int(num_steps / max_saved_states))
    
    # Initialize bodies
    # Body 0 is the Central Star (Sun)
    gms = [GM_SUN]
    pos = [np.array([0.0, 0.0, 0.0])]
    vel = [np.array([0.0, 0.0, 0.0])]
    names = ["Sun"]

    # Planetesimal distribution (power law Solid Surface Density Σ ~ r^-1.5)
    # Cumulative mass M(<r) ~ r^0.5
    # Draw radial distances r_i using inverse transform sampling
    np.random.seed(42) # Replicable initial disk configuration
    r_planets = (r_min**0.5 + np.random.rand(num_bodies - 1) * (r_max**0.5 - r_min**0.5))**2

    # Distribute masses: a few large embryos and a swarm of smaller planetesimals
    # Let's say 8% of bodies are larger planetary embryos (~1.0 Earth Mass)
    # The remaining are planetesimals
    num_embryos = int(np.ceil(0.08 * (num_bodies - 1)))
    num_planetesimals = (num_bodies - 1) - num_embryos

    # Total mass partition
    embryo_mass_total = num_embryos * 1.0 # Earth masses
    planetesimal_mass_total = total_disk_mass_earth - embryo_mass_total
    planetesimal_avg_mass = planetesimal_mass_total / num_planetesimals

    body_masses = []
    # Seed embryos
    for i in range(num_embryos):
        body_masses.append(1.0 + np.random.uniform(-0.2, 0.2)) # Earth masses
    # Seed planetesimals
    for i in range(num_planetesimals):
        body_masses.append(planetesimal_avg_mass * np.random.uniform(0.5, 1.5))
    
    np.random.shuffle(body_masses)

    # Assemble positions and velocities
    for i in range(num_bodies - 1):
        r_i = r_planets[i]
        m_earth = body_masses[i]
        m_solar = m_earth * M_EARTH_SOLAR
        gm_i = m_solar * GM_SUN

        # Angle around star
        theta = np.random.uniform(0, 2 * np.pi)

        # Position (with minor z inclination tilt)
        x = r_i * np.cos(theta)
        y = r_i * np.sin(theta)
        z = r_i * np.random.uniform(-0.015, 0.015)
        pos.append(np.array([x, y, z]))

        # Keplerian velocity magnitude v_K = sqrt(G*M_star / r)
        v_K = math.sqrt(GM_SUN / r_i)

        # Introduce a clean range of elliptical orbits
        if preset == "hot_jupiter":
            # Extreme eccentricity distribution (high end increased to 0.90)
            # Ensure pericenter r_p = r_i * (1 - e) > 0.06 AU to avoid starting inside the star
            max_e = min(0.90, 1.0 - 0.06 / r_i)
            e = np.random.uniform(0.02, max(0.02, max_e))
        else:
            # Solar system: moderate eccentricities (0.02 to 0.12)
            e = np.random.uniform(0.02, 0.12)

        is_pericenter = np.random.choice([True, False])
        ecc_factor = math.sqrt(1.0 + e) if is_pericenter else math.sqrt(1.0 - e)

        # Tangential velocity vector scaled for the elliptical orbit
        vx = -v_K * ecc_factor * np.sin(theta)
        vy = v_K * ecc_factor * np.cos(theta)
        vz = v_K * np.random.uniform(-0.02, 0.02) # Keep minor z-velocity for tilt
        vel.append(np.array([vx, vy, vz]))

        names.append(f"Planetesimal {i+1}")
        gms.append(gm_i)

    gms_arr = np.array(gms)
    pos_arr = np.array(pos)
    vel_arr = np.array(vel)

    # ----------------------------------------------------
    # Compile C Core Integrator
    # ----------------------------------------------------
    c_src = os.path.join(src_dir, "integrator.c")
    c_bin = os.path.join(src_dir, "integrator")
    
    print("🛠️ Compiling high-performance C core integrator...")
    compiled = False
    for compiler in ["clang", "gcc", "cc"]:
        try:
            subprocess.run([compiler, "-O3", c_src, "-o", c_bin, "-lm"], check=True)
            compiled = True
            print(f"✅ Compiled successfully using {compiler}")
            break
        except Exception:
            continue

    if not compiled:
        raise RuntimeError("❌ Failed to compile integrator.c. Please check your C compiler installation (clang/gcc).")

    # ----------------------------------------------------
    # Run Simulation
    # ----------------------------------------------------
    print(f"🚀 Running Early Solar System Simulation (Preset: {preset})")
    print(f"  Duration:          {years} years ({total_days} days)")
    print(f"  Step Size (dt):    {dt} days")
    print(f"  Total Steps:       {num_steps:,}")
    print(f"  Initial Bodies:    {num_bodies}")
    print(f"  Total Disk Mass:   {total_disk_mass_earth:.1f} M_earth")
    print(f"  Gas Eta (η):       {gas_eta}")
    print(f"  Base Drag Tau (τ): {tau_base} days")
    print(f"  Collision Inflation: {r_scale}x")

    input_bin = os.path.join(data_dir, "input.bin")
    output_bin = os.path.join(data_dir, "output.bin")
    collisions_csv = os.path.join(data_dir, f"collisions_{preset}.csv")

    # Write input binary file
    with open(input_bin, "wb") as f:
        f.write(gms_arr.tobytes())
        f.write(pos_arr.tobytes())
        f.write(vel_arr.tobytes())

    # Execute C program
    cmd = [
        c_bin,
        str(num_bodies),
        str(years),
        str(dt),
        str(save_every),
        str(r_scale),
        str(gas_eta),
        str(tau_base),
        str(drag_beta),
        str(GM_EARTH),
        input_bin,
        output_bin,
        collisions_csv
    ]
    
    subprocess.run(cmd, check=True)

    # ----------------------------------------------------
    # Read & Compress Simulation Results
    # ----------------------------------------------------
    print("💾 Reading simulation output binary...")
    step_size_doubles = 1 + num_bodies + num_bodies * 3 + num_bodies * 3
    data = np.fromfile(output_bin, dtype=np.float64)
    data = data.reshape(-1, step_size_doubles)

    saved_t = data[:, 0]
    saved_gms = data[:, 1 : 1 + num_bodies].reshape(-1, num_bodies)
    saved_r = data[:, 1 + num_bodies : 1 + num_bodies * 4].reshape(-1, num_bodies, 3)
    saved_v = data[:, 1 + num_bodies * 4 :].reshape(-1, num_bodies, 3)

    # Clean up temp binaries
    if os.path.exists(input_bin): os.remove(input_bin)
    if os.path.exists(output_bin): os.remove(output_bin)

    # Save to compressed .npz archive
    results_path = os.path.join(data_dir, f"simulation_{preset}.npz")
    np.savez_compressed(
        results_path,
        t=saved_t,
        gms=saved_gms,
        r=saved_r,
        v=saved_v,
        names=names,
        years=years,
        dt=dt,
        preset=preset
    )
    
    print(f"🎉 Simulation completed! Compressed results saved to: {results_path}")
    print(f"📂 Size: {os.path.getsize(results_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protoplanetary Disk N-body Accretion and Migration Simulator")
    parser.add_argument("--preset", type=str, default="solar_system", choices=["solar_system", "hot_jupiter"],
                        help="Presets for physical initial conditions (default: solar_system)")
    parser.add_argument("--years", type=float, default=800.0, help="Duration of simulation in Earth years (default: 800)")
    parser.add_argument("--dt", type=float, default=0.2, help="Timestep in days (default: 0.2)")
    args = parser.parse_args()

    run_simulation(preset=args.preset, years=args.years, dt=args.dt)
