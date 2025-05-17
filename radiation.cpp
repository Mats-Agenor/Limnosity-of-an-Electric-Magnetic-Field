#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <mpi.h>
#include <complex>
#include <fftw3-mpi.h>
#include <chrono>

// Physical constants
const double c = 3.0e8;          // Speed of light [m/s]
const double epsilon0 = 8.854e-12; // Vacuum permittivity [F/m]
const double mu0 = 4 * M_PI * 1e-7; // Vacuum permeability [N/A²]

struct FieldData {
    double t;
    double Ex, Ey, Ez;
    double Bx, By, Bz;
};

void read_field_data(const std::string& filename, std::vector<FieldData>& all_data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double x, y, z, q, Ex, Ey, Ez, Bx, By, Bz;
    int particle_count = 0;
    int time_step = 0;
    double dt = 1e-6; // Same as your simulation

    while (file >> x >> y >> z >> q >> Ex >> Ey >> Ez >> Bx >> By >> Bz) {
        if (particle_count == 0) {
            all_data.push_back({time_step * dt, Ex, Ey, Ez, Bx, By, Bz});
        }
        
        particle_count++;
        if (particle_count >= 10000) { // Same as your N particles
            particle_count = 0;
            time_step++;
            
            // Skip empty line between time steps
            std::string line;
            std::getline(file, line);
        }
    }
}

void calculate_power_spectrum(const std::vector<double>& signal, double dt, 
                             std::vector<double>& freq, std::vector<double>& power) {
    const ptrdiff_t N = signal.size();
    
    // Initialize FFTW
    fftw_plan plan;
    fftw_complex *in, *out;
    
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    
    // Prepare input data
    for (ptrdiff_t i = 0; i < N; ++i) {
        in[i][0] = signal[i]; // Real part
        in[i][1] = 0.0;       // Imaginary part
    }
    
    // Create plan and execute FFT
    plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    
    // Calculate power spectrum
    freq.resize(N/2);
    power.resize(N/2);
    
    for (ptrdiff_t i = 0; i < N/2; ++i) {
        freq[i] = i / (N * dt); // Frequency in Hz
        power[i] = (out[i][0]*out[i][0] + out[i][1]*out[i][1]) / N;
    }
    
    // Clean up
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // Read and distribute field data
    std::vector<FieldData> all_data;
    if (world_rank == 0) {
        read_field_data("trajectories_fields.dat", all_data);
        std::cout << "Read " << all_data.size() << " time steps of field data" << std::endl;
    }
    
    // Broadcast data size
    size_t data_size;
    if (world_rank == 0) {
        data_size = all_data.size();
    }
    MPI_Bcast(&data_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    // Resize on other processes
    if (world_rank != 0) {
        all_data.resize(data_size);
    }
    
    // Broadcast field data
    MPI_Bcast(all_data.data(), data_size * sizeof(FieldData), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Each process works on a portion of the data
    size_t chunk_size = data_size / world_size;
    size_t remainder = data_size % world_size;
    size_t start = world_rank * chunk_size + (world_rank < remainder ? world_rank : remainder);
    size_t end = start + chunk_size + (world_rank < remainder ? 1 : 0);
    
    std::cout << "Process " << world_rank << " handling time steps " 
              << start << " to " << end-1 << std::endl;
    
    // Extract Ex component for analysis (could use any field component)
    std::vector<double> Ex_signal;
    for (size_t i = start; i < end; ++i) {
        Ex_signal.push_back(all_data[i].Ex);
    }
    
    // Calculate power spectrum for this chunk
    std::vector<double> freq, power;
    double dt = 1e-6; // Same as your simulation
    calculate_power_spectrum(Ex_signal, dt, freq, power);
    
    // Gather results at root
    std::vector<double> all_freq, all_power;
    std::vector<int> recv_counts(world_size), displs(world_size);
    
    int local_size = freq.size();
    MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            displs[i] = displs[i-1] + recv_counts[i-1];
        }
        all_freq.resize(displs.back() + recv_counts.back());
        all_power.resize(displs.back() + recv_counts.back());
    }
    
    MPI_Gatherv(freq.data(), local_size, MPI_DOUBLE, 
               all_freq.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 
               0, MPI_COMM_WORLD);
    
    MPI_Gatherv(power.data(), local_size, MPI_DOUBLE, 
               all_power.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 
               0, MPI_COMM_WORLD);
    
    // Root process analyzes combined results
    if (world_rank == 0) {
        // Find dominant frequency
        double max_power = 0.0;
        size_t max_idx = 0;
        for (size_t i = 0; i < all_power.size(); ++i) {
            if (all_power[i] > max_power) {
                max_power = all_power[i];
                max_idx = i;
            }
        }
        
        double dominant_freq = all_freq[max_idx];
        double wavelength = c / dominant_freq;
        
        std::cout << "\n=== Radiation Analysis Results ===" << std::endl;
        std::cout << "Dominant frequency: " << dominant_freq << " Hz" << std::endl;
        std::cout << "Wavelength: " << wavelength << " meters" << std::endl;
        
        // Calculate total radiated power (using Larmor's formula approximation)
        // For a more accurate calculation, we'd need particle accelerations
        double total_power = 0.0;
        for (const auto& p : all_power) {
            total_power += p;
        }
        std::cout << "Total radiated power (relative): " << total_power << std::endl;
        
        // Output spectrum for plotting
        std::ofstream spec_file("power_spectrum.dat");
        for (size_t i = 0; i < all_freq.size(); ++i) {
            spec_file << all_freq[i] << " " << all_power[i] << "\n";
        }
        spec_file.close();
        std::cout << "Power spectrum saved to power_spectrum.dat" << std::endl;
        
        // Output wavelength vs time
        std::ofstream lambda_file("wavelength_vs_time.dat");
        for (size_t i = 0; i < all_data.size(); ++i) {
            // Simple approximation: wavelength from instantaneous Ex frequency
            double inst_freq = 1.0 / (2*M_PI) * sqrt(all_data[i].Ex*all_data[i].Ex + 
                              all_data[i].Ey*all_data[i].Ey + all_data[i].Ez*all_data[i].Ez);
            lambda_file << all_data[i].t << " " << c/inst_freq << "\n";
        }
        lambda_file.close();
        std::cout << "Wavelength vs time saved to wavelength_vs_time.dat" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
