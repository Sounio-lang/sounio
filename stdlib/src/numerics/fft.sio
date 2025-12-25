//! Fast Fourier Transform implementation

use std::math::{PI, cos, sin, sqrt}
use linalg::Vector

/// Complex number for FFT
#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }
    
    pub fn zero() -> Self {
        Complex { re: 0.0, im: 0.0 }
    }
    
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Complex {
            re: r * cos(theta),
            im: r * sin(theta),
        }
    }
    
    pub fn magnitude(&self) -> f64 {
        sqrt(self.re * self.re + self.im * self.im)
    }
    
    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }
    
    pub fn conj(&self) -> Self {
        Complex { re: self.re, im: -self.im }
    }
}

impl Add for Complex {
    type Output = Complex;
    
    fn add(self, other: Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl Sub for Complex {
    type Output = Complex;
    
    fn sub(self, other: Complex) -> Complex {
        Complex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl Mul for Complex {
    type Output = Complex;
    
    fn mul(self, other: Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl Mul<f64> for Complex {
    type Output = Complex;
    
    fn mul(self, scalar: f64) -> Complex {
        Complex {
            re: self.re * scalar,
            im: self.im * scalar,
        }
    }
}

/// FFT result
pub struct FFTResult {
    /// Transformed data
    pub data: Vec<Complex>,
    
    /// Whether the transform was successful
    pub success: bool,
    
    /// Error message if any
    pub message: string,
}

/// Cooley-Tukey FFT algorithm (radix-2)
pub fn fft(input: &[Complex]) -> FFTResult {
    let n = input.len();
    
    // Check if n is a power of 2
    if n == 0 || (n & (n - 1)) != 0 {
        return FFTResult {
            data: Vec::new(),
            success: false,
            message: "input length must be a power of 2".to_string(),
        };
    }
    
    let mut data = input.to_vec();
    
    // Bit-reversal permutation
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        
        if i < j {
            data.swap(i, j);
        }
    }
    
    // Cooley-Tukey FFT
    let mut length = 2;
    while length <= n {
        let wlen = Complex::from_polar(1.0, -2.0 * PI / length as f64);
        
        for i in (0..n).step_by(length) {
            let mut w = Complex::new(1.0, 0.0);
            
            for j in 0..length/2 {
                let u = data[i + j];
                let v = data[i + j + length/2] * w;
                
                data[i + j] = u + v;
                data[i + j + length/2] = u - v;
                
                w = w * wlen;
            }
        }
        
        length <<= 1;
    }
    
    FFTResult {
        data,
        success: true,
        message: "FFT completed successfully".to_string(),
    }
}

/// Inverse FFT
pub fn ifft(input: &[Complex]) -> FFTResult {
    let n = input.len();
    
    // Conjugate input
    let mut conj_input: Vec<Complex> = input.iter()
        .map(|c| c.conj())
        .collect();
    
    // Forward FFT
    let mut result = fft(&conj_input);
    
    if result.success {
        // Conjugate and scale
        for c in &mut result.data {
            *c = c.conj() * (1.0 / n as f64);
        }
    }
    
    result
}

/// Real FFT (for real-valued input)
pub fn rfft(input: &[f64]) -> FFTResult {
    let complex_input: Vec<Complex> = input.iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    
    fft(&complex_input)
}

/// Power spectral density
pub fn psd(input: &[f64], sample_rate: f64) -> (Vector<f64>, Vector<f64>) {
    let fft_result = rfft(input);
    let n = input.len();
    
    let mut frequencies = Vector::new(n / 2 + 1);
    let mut power = Vector::new(n / 2 + 1);
    
    for i in 0..=n/2 {
        frequencies[i] = i as f64 * sample_rate / n as f64;
        
        let magnitude = fft_result.data[i].magnitude();
        power[i] = magnitude * magnitude / (sample_rate * n as f64);
        
        // Account for negative frequencies (except DC and Nyquist)
        if i > 0 && i < n/2 {
            power[i] *= 2.0;
        }
    }
    
    (frequencies, power)
}

/// Convolution using FFT
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    
    // Find next power of 2
    let fft_size = n.next_power_of_two();
    
    // Zero-pad inputs
    let mut a_padded = vec![Complex::zero(); fft_size];
    let mut b_padded = vec![Complex::zero(); fft_size];
    
    for i in 0..a.len() {
        a_padded[i] = Complex::new(a[i], 0.0);
    }
    for i in 0..b.len() {
        b_padded[i] = Complex::new(b[i], 0.0);
    }
    
    // FFT both signals
    let fft_a = fft(&a_padded);
    let fft_b = fft(&b_padded);
    
    // Multiply in frequency domain
    let mut product = vec![Complex::zero(); fft_size];
    for i in 0..fft_size {
        product[i] = fft_a.data[i] * fft_b.data[i];
    }
    
    // Inverse FFT
    let ifft_result = ifft(&product);
    
    // Extract real part and trim to correct length
    ifft_result.data[..n].iter()
        .map(|c| c.re)
        .collect()
}
