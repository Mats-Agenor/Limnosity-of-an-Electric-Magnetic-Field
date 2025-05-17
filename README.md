# Limnosity-of-an-Electric-Magnetic-Field
---------------------
Compiling is complicated if you don't have the FFTW library installed, for me it worked with:

mpic++ -o radiation radiation.cpp -I$HOME/fftw-install/include -L$HOME/fftw-install/lib -lfftw3 -lfftw3_mpi -lm -O3 -std=c++17

Linking the already installed library that I use for GADGET-4

To run, use: mpirun -np 20 ./radiation

The code will generate two output .dat files (wavelength_vs_time.dat and power_spectrum.dat) that will be used to plot the visualizations.

---------------------
This C++ program is based on the program in the "Electric Magnetic Fields" repository and is designed to analyze electromagnetic field data from particle simulations. The code reads field data, calculates power spectra using Fourier transforms, and analyzes radiation characteristics, all using parallel processing with MPI.

The code begins by defining physical constants and a data structure to store field information. The `read_field_data` function reads simulation data from a file, storing electric and magnetic field components at each time step. The main analysis occurs in `calculate_power_spectrum`, which uses FFTW to calculate the frequency spectrum of the field signals.

The importance of the FFTW library cannot be overstated in this context. FFTW (Fastest Fourier Transform in the West) is crucial because it provides highly optimized implementations of the Fast Fourier Transform (FFT), which is essential for converting time-domain field data into frequency-domain information. The FFT allows us to identify dominant frequencies in electromagnetic radiation, which would be computationally prohibitive to calculate directly in the time domain. The MPI-enabled version of FFTW (fftw3-mpi) is particularly valuable here, as it allows distributed computation of large transforms across multiple processes, essential for dealing with potentially massive datasets from particle simulations.

The code uses the Fourier transform to decompose electromagnetic field signals into their frequency components. This makes physical sense because accelerated charged particles emit radiation across a spectrum of frequencies, and the Fourier transform naturally reveals these spectral features. Computing the power spectrum (squared magnitude of the Fourier coefficients) provides the distribution of energy across frequencies, allowing identification of the dominant radiation frequencies.

The parallel implementation divides the data analysis between MPI processes, with each process handling a portion of the time series data. This parallel approach is justified because FFT calculations are computationally intensive and can be performed independently on different data segments. The root process then combines the results to determine the overall characteristics of the radiation, including the dominant frequency and wavelength.

The physics behind the analysis are based on classical principles of electromagnetics. The code approximates the properties of the radiation using field data, with the dominant frequency corresponding to the peak in the power spectrum. The wavelength is then derived from this frequency using the speed of light ratio. Although the code includes a simplified approximation of the instantaneous frequency (based on the field magnitude), a more rigorous analysis would require particle acceleration data for accurate radiation calculations.

The output includes spectral information and the time evolution of the wavelength, providing insights into how the radiation characteristics change during the simulation. This type of analysis is particularly valuable for understanding the physics of particle accelerators, plasma radiation, and other scenarios in which charged particles emit electromagnetic waves.
