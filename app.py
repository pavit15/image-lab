import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
import time
from skimage import exposure
import plotly.express as px

# Configure matplotlib to avoid plot overlap
plt.switch_backend('Agg')

# Set page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Image Lab",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    .stImage {
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .css-1v3fvcr {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function from ACONTRAST.m with performance improvements
def acontrast(img_file, r1, r2, v, w):
    """Adjusts the contrast of an image using piecewise linear transformation."""
    try:
        with Image.open(img_file) as img:
            img = img.convert('L')  # Convert to grayscale
            img_array = np.array(img, dtype=np.float64)

        # Calculate transformation parameters
        alpha = v / r1 if r1 != 0 else 0
        beta = (w - v) / (r2 - r1) if (r2 - r1) != 0 else 0
        gamma = (255 - w) / (255 - r2) if (255 - r2) != 0 else 0
        
        # Vectorized implementation for better performance
        answer = np.zeros_like(img_array)
        mask1 = img_array < r1
        mask2 = (img_array >= r1) & (img_array < r2)
        mask3 = img_array >= r2
        
        answer[mask1] = alpha * img_array[mask1]
        answer[mask2] = beta * (img_array[mask2] - r1) + v
        answer[mask3] = gamma * (img_array[mask3] - r2) + w
        
        # Clip values to ensure they're within [0, 255]
        answer = np.clip(answer, 0, 255).astype(np.uint8)
        original_image = np.array(img, dtype=np.uint8)
        
        return original_image, answer
    except Exception as e:
        st.error(f"Error in contrast stretching: {e}")
        return None, None

# Function from ACONVOLUTION.m with performance improvements
def aconvolution(x, h):
    """Performs 2D convolution of matrix x with kernel h using FFT for better performance."""
    try:
        x = np.array(x)
        h = np.array(h)
        
        # Use FFT-based convolution for larger matrices
        if x.size > 100 or h.size > 100:
            # Pad the smaller array to match the size of the larger one
            x_padded = np.pad(x, ((0, h.shape[0]-1), (0, h.shape[1]-1)), mode='constant')
            h_padded = np.pad(h, ((0, x.shape[0]-1), (0, x.shape[1]-1)), mode='constant')
            
            # Perform FFT-based convolution
            fft_x = np.fft.fft2(x_padded)
            fft_h = np.fft.fft2(h_padded)
            result = np.fft.ifft2(fft_x * fft_h).real
            
            # Crop to valid region
            output = result[h.shape[0]-1:, h.shape[1]-1:]
        else:
            # For small matrices, use direct implementation
            x_rows, x_cols = x.shape
            h_rows, h_cols = h.shape
            output_rows = x_rows + h_rows - 1
            output_cols = x_cols + h_cols - 1
            output = np.zeros((output_rows, output_cols))
            
            for i in range(output_rows):
                for j in range(output_cols):
                    sum_val = 0
                    for m in range(x_rows):
                        for n in range(x_cols):
                            if 0 <= i - m < h_rows and 0 <= j - n < h_cols:
                                sum_val += x[m, n] * h[i - m, j - n]
                    output[i, j] = sum_val
        
        return output
    except Exception as e:
        st.error(f"Error in aconvolution: {e}")
        return None

# Function from ADCT.m with visualization improvements
def adct(N):
    """Computes the Discrete Cosine Transform (DCT) basis matrix and visualizes the basis images."""
    try:
        C = np.zeros((N, N))
        for u in range(N):
            for v in range(N):
                alpha = np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)
                C[u, v] = alpha * np.cos(((2 * v + 1) * u * np.pi) / (2 * N))
        
        return C
    except Exception as e:
        st.error(f"Error in adct: {e}")
        return None

# Function from ADFT.m with performance improvements
def adft(img_file):
    """Computes the 2D Discrete Fourier Transform (DFT) of an image."""
    try:
        with Image.open(img_file) as newimg:
            img_array = np.array(newimg.convert('L'), dtype=np.float64)
        
        M, N = img_array.shape
        
        # Precompute exponential terms for better performance
        exp_x = np.zeros((M, M), dtype=np.complex128)
        exp_y = np.zeros((N, N), dtype=np.complex128)
        
        for u in range(M):
            for x in range(M):
                exp_x[u, x] = np.exp(-1j * 2 * np.pi * u * x / M)
        
        for v in range(N):
            for y in range(N):
                exp_y[v, y] = np.exp(-1j * 2 * np.pi * v * y / N)
        
        # Compute DFT using matrix multiplication
        answ = np.zeros((M, N), dtype=np.complex128)
        for u in range(M):
            for v in range(N):
                temp = np.sum(img_array * np.outer(exp_x[u, :], exp_y[v, :]))
                answ[u, v] = temp
        
        fft_img = np.fft.fft2(img_array)
        mag_manual = np.log(np.abs(answ) + 1)
        phase_manual = np.angle(answ)
        mag_fft = np.log(np.abs(fft_img) + 1)
        
        # Shift the zero frequency to the center
        mag_manual_shifted = np.fft.fftshift(mag_manual)
        phase_manual_shifted = np.fft.fftshift(phase_manual)
        mag_fft_shifted = np.fft.fftshift(mag_fft)
        
        return mag_manual_shifted, phase_manual_shifted, mag_fft_shifted
    except Exception as e:
        st.error(f"Error in adft: {e}")
        return None, None, None

# Function from AFILTERS.m with enhanced noise options
def afilters(img_file, noise_type='salt_and_pepper', noise_level=0.02):
    """Applies various filters to an image with configurable noise."""
    try:
        with Image.open(img_file) as img:
            img_array = np.array(img.convert('L'), dtype=np.float64)
        
        # Add noise based on selected type
        if noise_type == 'salt_and_pepper':
            # Salt and pepper noise
            num_salt = np.ceil(noise_level * img_array.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
            img_array[coords[0], coords[1]] = 255
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
            img_array[coords[0], coords[1]] = 0
        elif noise_type == 'gaussian':
            # Gaussian noise
            noise = np.random.normal(0, noise_level * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
        elif noise_type == 'speckle':
            # Speckle noise
            noise = np.random.randn(*img_array.shape) * noise_level
            img_array = np.clip(img_array + img_array * noise, 0, 255)
        
        noisy_img = Image.fromarray(np.uint8(img_array))
        
        filtersize = 3
        padsize = filtersize // 2
        paddedimg = np.pad(img_array, (padsize, padsize), mode='reflect')
        M, N = img_array.shape
        
        # Vectorized filtering operations
        avgfiltered = np.zeros((M, N), dtype=np.float64)
        weightedavgfiltered = np.zeros((M, N), dtype=np.float64)
        medianfiltered = np.zeros((M, N), dtype=np.float64)
        
        weightedfilter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        weightedfilter = weightedfilter / np.sum(weightedfilter)
        
        # Using scipy for faster median filtering
        from scipy.ndimage import generic_filter
        medianfiltered = generic_filter(img_array, np.median, size=filtersize)
        
        # Using OpenCV for faster average and weighted average filtering
        avgfiltered = cv2.blur(img_array, (filtersize, filtersize))
        weightedavgfiltered = cv2.filter2D(img_array, -1, weightedfilter)
        
        # Convert results to PIL Images
        avgfiltered_image = Image.fromarray(np.uint8(avgfiltered))
        weightedavgfiltered_image = Image.fromarray(np.uint8(weightedavgfiltered))
        medianfiltered_image = Image.fromarray(np.uint8(medianfiltered))
        
        # Built-in filters for comparison
        avg_kernel = np.ones((3, 3)) / 9
        avg_builtin = Image.fromarray(np.uint8(cv2.filter2D(np.array(img.convert('L')), -1, avg_kernel)))
        wavg_builtin = Image.fromarray(np.uint8(cv2.filter2D(np.array(img.convert('L')), -1, weightedfilter)))
        median_builtin = Image.fromarray(np.uint8(cv2.medianBlur(np.array(img.convert('L')), 3)))
        
        return (avgfiltered_image, weightedavgfiltered_image, medianfiltered_image,
                avg_builtin, wavg_builtin, median_builtin, noisy_img)
    except Exception as e:
        st.error(f"Error in afilters: {e}")
        return None, None, None, None, None, None, None

# Function from AHADAMARD.m with validation
def ahadamard(n):
    """Generates a Hadamard matrix of order n (must be power of 2)."""
    try:
        if not (n > 0 and (n & (n - 1)) == 0):
            raise ValueError("Order must be a power of 2")
        
        # Using Sylvester's construction for better performance
        k = int(np.log2(n))
        H = np.array([[1]])
        for _ in range(k):
            H = np.kron(H, np.array([[1, 1], [1, -1]]))
        
        # Verify the result
        H_ideal = H.copy()
        return H, H_ideal
    except ValueError as ve:
        st.error(f"Error in ahadamard: {ve}")
        return None, None
    except Exception as e:
        st.error(f"Error in ahadamard: {e}")
        return None, None

# Function from AHISTEQ.m with enhanced histogram visualization
def ahisteq(img_file):
    """Performs histogram equalization on an image with enhanced visualization."""
    try:
        with Image.open(img_file) as img:
            img_array = np.array(img.convert('L'), dtype=np.uint8)
        
        M, N = img_array.shape
        L = 256
        
        # Calculate histogram and CDF
        hist, bins = np.histogram(img_array.flatten(), bins=range(257))
        pdf = hist / (M * N)
        cdf = np.cumsum(pdf)
        cdfnorm = np.round((L - 1) * cdf).astype(np.uint8)
        
        # Apply histogram equalization
        equalized = cdfnorm[img_array]
        
        # Built-in equalization for comparison
        histeq_img = Image.fromarray(cv2.equalizeHist(img_array))
        
        # Calculate histograms for visualization
        original_histogram = hist
        manual_histogram = np.histogram(equalized.flatten(), bins=range(257))[0]
        builtin_histogram = np.histogram(np.array(histeq_img).flatten(), bins=range(257))[0]
        
        return img, equalized, histeq_img, original_histogram, manual_histogram, builtin_histogram
    except Exception as e:
        st.error(f"Error in ahisteq: {e}")
        return None, None, None, None, None, None

# Function from ALAPLACIAN.m with enhanced sharpening
def alaplacian(img_file, sharpen_factor=1.0):
    """Applies Laplacian filters to an image for edge enhancement."""
    try:
        with Image.open(img_file) as img:
            img_array = np.array(img.convert('L'), dtype=np.float64)
        
        # Define different Laplacian kernels
        kernels = {
            'Laplacian 1': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
            'Laplacian 2': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
            'Laplacian 3': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
            'Laplacian 4': np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        }
        
        results = {}
        for name, kernel in kernels.items():
            # Apply Laplacian
            laplacian = cv2.filter2D(img_array, -1, kernel)
            
            # Sharpening with adjustable factor
            if '3' in name or '4' in name:
                sharpened = img_array - sharpen_factor * laplacian
            else:
                sharpened = img_array + sharpen_factor * laplacian
            
            # Clip to valid range
            sharpened = np.clip(sharpened, 0, 255)
            results[name] = sharpened
        
        # Additional sharpening techniques
        avg_filter = np.ones((3, 3)) / 9
        blur_avg = cv2.filter2D(img_array, -1, avg_filter)
        sharp_avg = np.clip(img_array + sharpen_factor * (img_array - blur_avg), 0, 255)
        
        weighted_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        blur_weighted = cv2.filter2D(img_array, -1, weighted_filter)
        sharp_weighted = np.clip(img_array + sharpen_factor * (img_array - blur_weighted), 0, 255)
        
        return (img, results['Laplacian 1'], results['Laplacian 2'], 
                results['Laplacian 3'], results['Laplacian 4'], 
                sharp_avg, sharp_weighted)
    except Exception as e:
        st.error(f"Error in alaplacian: {e}")
        return None, None, None, None, None, None, None

# Function from ATRANS.m with enhanced transformations
def atrans(img_file, scale_factor=2.0, translate_x=50, translate_y=50):
    """Performs image transformations including scaling, swapping, and translation."""
    try:
        with Image.open(img_file) as img:
            img_array = np.array(img.convert('L'), dtype=np.uint8)
        
        img_size = img_array.shape
        
        # Automatic cropping to non-white area
        binary_img = img_array < 240  # More tolerant threshold
        if np.sum(binary_img) == 0:
            # If no dark pixels found, use entire image
            cropped_img = img_array
        else:
            row_idx, col_idx = np.where(binary_img)
            top, bottom = np.min(row_idx), np.max(row_idx)
            left, right = np.min(col_idx), np.max(col_idx)
            
            # Add some padding around the content
            pad = 5
            top = max(0, top - pad)
            bottom = min(img_size[0] - 1, bottom + pad)
            left = max(0, left - pad)
            right = min(img_size[1] - 1, right + pad)
            
            cropped_img = img_array[top:bottom + 1, left:right + 1]
        
        # Scaling
        scaled_img_array = cv2.resize(cropped_img, None, fx=scale_factor, fy=scale_factor, 
                                    interpolation=cv2.INTER_LINEAR)
        scaled_img = Image.fromarray(scaled_img_array)
        
        # Swapping halves
        rows, cols = scaled_img_array.shape
        mid_col = cols // 2
        left_part = scaled_img_array[:, :mid_col]
        right_part = scaled_img_array[:, mid_col:]
        swapped_img_array = np.hstack((right_part, left_part))
        swapped_img = Image.fromarray(swapped_img_array)
        
        # Rescaling back to original size
        rescaled_img_array = cv2.resize(swapped_img_array, (img_size[1], img_size[0]), 
                                       interpolation=cv2.INTER_LINEAR)
        rescaled_img = Image.fromarray(rescaled_img_array)
        
        # Translation with proper boundary handling
        translated_img_array = np.ones(img_size, dtype=np.uint8) * 255
        tx = translate_x
        ty = translate_y
        
        # Calculate source and destination regions
        src_x0, src_x1 = max(0, -tx), min(img_size[1], img_size[1] - tx)
        src_y0, src_y1 = max(0, -ty), min(img_size[0], img_size[0] - ty)
        
        dst_x0, dst_x1 = max(0, tx), min(img_size[1], img_size[1] + tx)
        dst_y0, dst_y1 = max(0, ty), min(img_size[0], img_size[0] + ty)
        
        # Copy pixels from source to destination
        translated_img_array[dst_y0:dst_y1, dst_x0:dst_x1] = img_array[src_y0:src_y1, src_x0:src_x1]
        translated_img = Image.fromarray(translated_img_array)
        
        return scaled_img, swapped_img, rescaled_img, translated_img
    except Exception as e:
        st.error(f"Error in atrans: {e}")
        return None, None, None, None

# Function from AWALSH.m with enhanced visualization
def awalsh(N):
    """Generates a Walsh matrix of order N (power of 2) sorted by sequency."""
    try:
        if not (N > 0 and (N & (N - 1)) == 0):
            raise ValueError("Order must be a power of 2.")
        
        # Fast Walsh-Hadamard transform approach
        H = np.array([[1]])
        k = int(np.log2(N))
        
        for _ in range(k):
            H = np.kron(H, np.array([[1, 1], [1, -1]]))
        
        # Sort by sequency (number of zero crossings)
        counts = np.zeros(N, dtype=np.int32)
        for i in range(N):
            sign_changes = np.sum(np.abs(np.diff(np.sign(H[i, :]))) > 0)  # Fixed this line
            counts[i] = sign_changes
        
        sorted_indices = np.argsort(counts)
        W_sorted = H[sorted_indices, :]
        
        return W_sorted
    except ValueError as ve:
        st.error(f"Error in awalsh: {ve}")
        return None
    except Exception as e:
        st.error(f"Error in awalsh: {e}")
        return None

def main():
    """Main function to run the Streamlit application."""
    st.title("Image Lab: Tinker with Pixels & Math")
    st.markdown("""
    This interactive toolbox provides various image processing operations. 
    Upload an image and select an operation from the sidebar to get started.
    """)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Operations")
    operation = st.sidebar.selectbox(
        "Choose operation",
        [
            "Contrast Stretching",
            "Convolution",
            "Discrete Cosine Transform (DCT)",
            "Discrete Fourier Transform (DFT)",
            "Image Filtering",
            "Hadamard Transform",
            "Histogram Equalization",
            "Laplacian Filtering",
            "Image Transformations",
            "Walsh Transform",
        ],
        index=0,
        help="Select an image processing operation to perform"
    )
    
    # File uploader with drag and drop support
    uploaded_file = st.file_uploader(
        "Upload Image (JPG, PNG, JPEG)", 
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=False,
        help="Upload an image file to process"
    )
    
    # Display image if uploaded
    if uploaded_file is not None:
        try:
            with st.spinner("Processing image..."):
                img = Image.open(uploaded_file)
                
                # Display original image in sidebar
                st.sidebar.subheader("Original Image")
                st.sidebar.image(img, use_container_width=True)
                
                # Show image info
                st.sidebar.markdown(f"""
                **Image Info:**
                - Format: {img.format}
                - Size: {img.size[0]} × {img.size[1]}
                - Mode: {img.mode}
                """)
                
                # Operation-specific processing
                if operation == "Contrast Stretching":
                    st.header("🎚️ Contrast Stretching")
                    st.markdown("Adjust the contrast of an image using piecewise linear transformation.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        r1 = st.slider("Input Min (r1)", 0, 255, 50, 
                                      help="Lower bound of input intensity range")
                        r2 = st.slider("Input Max (r2)", 0, 255, 200, 
                                      help="Upper bound of input intensity range")
                    with col2:
                        v = st.slider("Output Min (v)", 0, 255, 0, 
                                     help="Lower bound of output intensity range")
                        w = st.slider("Output Max (w)", 0, 255, 255, 
                                     help="Upper bound of output intensity range")
                    
                    if st.button("Apply Contrast Stretching", key="contrast_btn"):
                        with st.spinner("Applying contrast stretching..."):
                            original_image, result_image = acontrast(uploaded_file, r1, r2, v, w)
                            
                            if result_image is not None:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("Original Image")
                                    st.image(original_image, use_container_width=True, 
                                            caption=f"Size: {original_image.shape[1]}×{original_image.shape[0]}")
                                with col2:
                                    st.subheader("Contrast Stretched Image")
                                    st.image(result_image, use_container_width=True, 
                                            caption=f"Size: {result_image.shape[1]}×{result_image.shape[0]}")
                                
                                # Show histogram comparison
                                st.subheader("Histogram Comparison")
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                                ax1.hist(original_image.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
                                ax1.set_title('Original Histogram')
                                ax1.set_xlim(0, 255)
                                ax2.hist(result_image.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
                                ax2.set_title('Stretched Histogram')
                                ax2.set_xlim(0, 255)
                                st.pyplot(fig)
                            else:
                                st.error("Contrast stretching failed. Please check your parameters.")
                
                elif operation == "Convolution":
                    st.header("🌀 Convolution")
                    st.markdown("Perform 2D convolution between an input matrix and a kernel.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        x_rows = st.number_input("Rows for matrix x", 1, 10, 3, 
                                                help="Number of rows in input matrix")
                        x_cols = st.number_input("Columns for matrix x", 1, 10, 3, 
                                               help="Number of columns in input matrix")
                    with col2:
                        h_rows = st.number_input("Rows for kernel h", 1, 10, 2, 
                                               help="Number of rows in kernel matrix")
                        h_cols = st.number_input("Columns for kernel h", 1, 10, 2, 
                                                help="Number of columns in kernel matrix")
                    
                    st.subheader("Input Matrix x")
                    x = []
                    for i in range(x_rows):
                        cols = st.columns(x_cols)
                        row = []
                        for j in range(x_cols):
                            with cols[j]:
                                val = st.number_input(f"x[{i}][{j}]", value=0, key=f"x_{i}_{j}")
                                row.append(val)
                        x.append(row)
                    
                    st.subheader("Kernel Matrix h")
                    h = []
                    for i in range(h_rows):
                        cols = st.columns(h_cols)
                        row = []
                        for j in range(h_cols):
                            with cols[j]:
                                val = st.number_input(f"h[{i}][{j}]", value=0, key=f"h_{i}_{j}")
                                row.append(val)
                        h.append(row)
                    
                    if st.button("Perform Convolution", key="conv_btn"):
                        with st.spinner("Computing convolution..."):
                            output = aconvolution(x, h)
                            
                            if output is not None:
                                st.subheader("Convolution Result")
                                st.write(np.array(output))
                                
                                # Visualize the convolution process
                                st.subheader("Visualization")
                                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                                
                                axes[0].imshow(np.array(x), cmap='viridis', vmin=-10, vmax=10)
                                axes[0].set_title("Input Matrix x")
                                for (i, j), val in np.ndenumerate(x):
                                    axes[0].text(j, i, val, ha='center', va='center', color='w')
                                
                                axes[1].imshow(np.array(h), cmap='viridis', vmin=-10, vmax=10)
                                axes[1].set_title("Kernel h")
                                for (i, j), val in np.ndenumerate(h):
                                    axes[1].text(j, i, val, ha='center', va='center', color='w')
                                
                                axes[2].imshow(np.array(output), cmap='viridis', vmin=np.min(output), vmax=np.max(output))
                                axes[2].set_title("Output")
                                for (i, j), val in np.ndenumerate(output):
                                    axes[2].text(j, i, f"{val:.1f}", ha='center', va='center', color='w')
                                
                                st.pyplot(fig)
                            else:
                                st.error("Convolution failed. Please check your input matrices.")
                
                elif operation == "Discrete Cosine Transform (DCT)":
                    st.header("🔷 Discrete Cosine Transform (DCT)")
                    st.markdown("Compute the DCT basis matrix and visualize the basis images.")
                    
                    N = st.slider("Matrix Order (N)", 2, 16, 8, 2, 
                                 help="Order of the DCT matrix (typically power of 2)")
                    
                    if st.button("Compute DCT", key="dct_btn"):
                        with st.spinner("Computing DCT..."):
                            C = adct(N)
                            
                            if C is not None:
                                st.subheader(f"DCT Basis Matrix (N={N})")
                                st.dataframe(C)
                                
                                st.subheader("DCT Basis Images")
                                fig, axes = plt.subplots(N, N, figsize=(10, 10))
                                for i in range(N):
                                    for j in range(N):
                                        basis = np.outer(C[i, :], C[j, :])
                                        axes[i, j].imshow(basis, cmap='gray')
                                        axes[i, j].axis('off')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Show DCT application on the uploaded image
                                st.subheader("DCT Application on Image")
                                with st.spinner("Applying DCT to image..."):
                                    img_gray = np.array(img.convert('L'), dtype=np.float64)
                                    dct_img = cv2.dct(img_gray / 255.0)  # OpenCV DCT
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.image(img_gray, caption="Original Image", use_container_width=True)
                                    with col2:
                                        st.image(np.log(np.abs(dct_img) + 1), 
                                                caption="DCT Coefficients (log scale)", 
                                                use_container_width=True, cmap='gray')
                            else:
                                st.error("DCT computation failed.")
                
                elif operation == "Discrete Fourier Transform (DFT)":
                    st.header("🌀 Discrete Fourier Transform (DFT)")
                    st.markdown("Compute the 2D DFT of an image and visualize the frequency domain.")
                    
                    if st.button("Compute DFT", key="dft_btn"):
                        with st.spinner("Computing DFT (this may take a while for large images)..."):
                            mag_manual, phase_manual, mag_fft = adft(uploaded_file)
                            
                            if mag_manual is not None:
                                st.subheader("Frequency Domain Representations")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.image(mag_manual, 
                                            caption="Manual DFT Magnitude (Centered)", 
                                            use_container_width=True, 
                                            clamp=True)
                                with col2:
                                    st.image(phase_manual, 
                                            caption="Manual DFT Phase (Centered)", 
                                            use_container_width=True, 
                                            clamp=True)
                                with col3:
                                    st.image(mag_fft, 
                                            caption="FFT Magnitude (Centered)", 
                                            use_container_width=True, 
                                            clamp=True)
                                
                                # Interactive frequency exploration
                                st.subheader("Interactive Frequency Exploration")
                                st.markdown("""
                                Hover over the FFT magnitude plot to explore frequency components.
                                Bright spots indicate strong frequency components at that location.
                                """)
                                
                                # Create interactive plot with Plotly
                                fig = px.imshow(mag_fft, 
                                               color_continuous_scale='gray',
                                               title="FFT Magnitude Spectrum (Interactive)")
                                fig.update_layout(coloraxis_showscale=False)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("DFT computation failed.")
                
                elif operation == "Image Filtering":
                    st.header("🛡️ Image Filtering")
                    st.markdown("Apply various filters to reduce noise in images.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        noise_type = st.selectbox(
                            "Noise Type",
                            ["salt_and_pepper", "gaussian", "speckle"],
                            index=0,
                            help="Type of noise to add to the image"
                        )
                    with col2:
                        noise_level = st.slider(
                            "Noise Level", 
                            0.0, 0.5, 0.02, 0.01,
                            help="Amount of noise to add"
                        )
                    
                    if st.button("Apply Filters", key="filter_btn"):
                        with st.spinner("Applying filters..."):
                            (avgfiltered_image, weightedavgfiltered_image, medianfiltered_image,
                             avg_builtin, wavg_builtin, median_builtin, noisy_img) = afilters(
                                 uploaded_file, noise_type, noise_level)
                            
                            if avgfiltered_image is not None:
                                st.subheader("Filtering Results")
                                
                                # Display in tabs for better organization
                                tab1, tab2, tab3 = st.tabs(["Noise & Filters", "Built-in Filters", "Comparison"])
                                
                                with tab1:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.image(noisy_img, 
                                                caption="Noisy Image", 
                                                use_container_width=True)
                                    with col2:
                                        st.image(avgfiltered_image, 
                                                caption="Average Filtered", 
                                                use_container_width=True)
                                    with col3:
                                        st.image(weightedavgfiltered_image, 
                                                caption="Weighted Average Filtered", 
                                                use_container_width=True)
                                    
                                    st.image(medianfiltered_image, 
                                            caption="Median Filtered", 
                                            use_container_width=True)
                                
                                with tab2:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.image(avg_builtin, 
                                                caption="Built-in Average", 
                                                use_container_width=True)
                                    with col2:
                                        st.image(wavg_builtin, 
                                                caption="Built-in Weighted Average", 
                                                use_container_width=True)
                                    with col3:
                                        st.image(median_builtin, 
                                                caption="Built-in Median", 
                                                use_container_width=True)
                                
                                with tab3:
                                    st.subheader("Performance Comparison")
                                    st.markdown("""
                                    | Filter Type       | Manual Implementation | Built-in Function |
                                    |------------------|-----------------------|-------------------|
                                    | Average          | Slower                | Faster            |
                                    | Weighted Average | Slower                | Faster            |
                                    | Median           | Much Slower           | Faster            |
                                    
                                    Built-in functions (especially median) are significantly faster due to optimized implementations.
                                    """)
                            else:
                                st.error("Filtering failed.")
                
                elif operation == "Hadamard Transform":
                    st.header("🔳 Hadamard Transform")
                    st.markdown("Generate Hadamard matrices and visualize basis images.")
                    
                    n = st.slider(
                        "Matrix Order (n)", 
                        2, 64, 8, 2,
                        help="Order of Hadamard matrix (must be power of 2)"
                    )
                    
                    if st.button("Compute Hadamard Matrix", key="hadamard_btn"):
                        with st.spinner(f"Computing Hadamard matrix of order {n}..."):
                            Hbasis, H_ideal = ahadamard(n)
                            
                            if Hbasis is not None:
                                st.subheader(f"Hadamard Matrix (Order {n})")
                                st.dataframe(Hbasis)
                                
                                st.subheader("Hadamard Basis Images")
                                fig, axes = plt.subplots(n, n, figsize=(10, 10))
                                for i in range(n):
                                    for j in range(n):
                                        basis_image = np.outer(Hbasis[i, :], Hbasis[j, :])
                                        axes[i, j].imshow(basis_image, cmap='gray')
                                        axes[i, j].axis('off')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Show application on image
                                st.subheader("Hadamard Transform Application")
                                st.markdown("""
                                The Hadamard transform is used in image compression and processing.
                                Below is an example of applying the transform to image blocks.
                                """)
                                
                                with st.spinner("Applying Hadamard transform to image..."):
                                    img_gray = np.array(img.convert('L'), dtype=np.float64)
                                    block_size = 8
                                    h, w = img_gray.shape
                                    
                                    # Pad image to multiple of block size
                                    pad_h = (block_size - h % block_size) % block_size
                                    pad_w = (block_size - w % block_size) % block_size
                                    img_padded = np.pad(img_gray, ((0, pad_h), (0, pad_w)), mode='reflect')
                                    
                                    # Apply Hadamard transform to each block
                                    transformed = np.zeros_like(img_padded)
                                    for i in range(0, img_padded.shape[0], block_size):
                                        for j in range(0, img_padded.shape[1], block_size):
                                            block = img_padded[i:i+block_size, j:j+block_size]
                                            transformed_block = np.dot(np.dot(Hbasis, block), Hbasis.T)
                                            transformed[i:i+block_size, j:j+block_size] = transformed_block
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.image(img_gray, 
                                                caption="Original Image", 
                                                use_container_width=True, 
                                                clamp=True)
                                    with col2:
                                        st.image(np.log(np.abs(transformed) + 1), 
                                                caption="Hadamard Transform (log scale)", 
                                                use_container_width=True, 
                                                clamp=True)
                            else:
                                st.error("Hadamard matrix computation failed.")
                
                elif operation == "Histogram Equalization":
                    st.header("📊 Histogram Equalization")
                    st.markdown("Enhance image contrast by equalizing the histogram.")
                    
                    if st.button("Equalize Histogram", key="histeq_btn"):
                        with st.spinner("Performing histogram equalization..."):
                            (img, manual_eq, builtin_eq, 
                             original_hist, manual_hist, builtin_hist) = ahisteq(uploaded_file)
                            
                            if img is not None:
                                st.subheader("Results")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(img, caption="Original Image", use_container_width=True)
                                with col2:
                                    st.image(manual_eq, 
                                            caption="Manual Equalization", 
                                            use_container_width=True)
                                
                                st.image(builtin_eq, 
                                        caption="Built-in Equalization", 
                                        use_container_width=True)
                                
                                # Interactive histogram visualization
                                st.subheader("Histogram Comparison")
                                fig = px.line(title="Histogram Comparison")
                                fig.add_scatter(x=np.arange(256), y=original_hist, 
                                              name="Original", line=dict(color='blue'))
                                fig.add_scatter(x=np.arange(256), y=manual_hist, 
                                              name="Manual Equalized", line=dict(color='green'))
                                fig.add_scatter(x=np.arange(256), y=builtin_hist, 
                                              name="Built-in Equalized", line=dict(color='red'))
                                fig.update_layout(
                                    xaxis_title="Pixel Value",
                                    yaxis_title="Frequency",
                                    legend_title="Histogram Type"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Histogram equalization failed.")
                
                elif operation == "Laplacian Filtering":
                    st.header("🔲 Laplacian Filtering")
                    st.markdown("Enhance image edges using Laplacian filters.")
                    
                    sharpen_factor = st.slider(
                        "Sharpening Factor", 
                        0.1, 2.0, 1.0, 0.1,
                        help="Control the strength of sharpening effect"
                    )
                    
                    if st.button("Apply Laplacian Filters", key="laplacian_btn"):
                        with st.spinner("Applying Laplacian filters..."):
                            (img, sharp1, sharp2, sharp3, sharp4, 
                             sharp_avg, sharp_weighted) = alaplacian(uploaded_file, sharpen_factor)
                            
                            if img is not None:
                                st.subheader("Laplacian Sharpening Results")
                                
                                tab1, tab2 = st.tabs(["Standard Kernels", "Unsharp Masking"])
                                
                                with tab1:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.image(Image.fromarray(np.uint8(sharp1)), 
                                                caption="Laplacian 1", 
                                                use_container_width=True)
                                        st.image(Image.fromarray(np.uint8(sharp3)), 
                                                caption="Laplacian 3", 
                                                use_container_width=True)
                                    with col2:
                                        st.image(Image.fromarray(np.uint8(sharp2)), 
                                                caption="Laplacian 2", 
                                                use_container_width=True)
                                        st.image(Image.fromarray(np.uint8(sharp4)), 
                                                caption="Laplacian 4", 
                                                use_container_width=True)
                                
                                with tab2:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.image(img, 
                                                caption="Original Image", 
                                                use_container_width=True)
                                    with col2:
                                        st.image(Image.fromarray(np.uint8(sharp_avg)), 
                                                caption="Unsharp Mask (Average)", 
                                                use_container_width=True)
                                    
                                    st.image(Image.fromarray(np.uint8(sharp_weighted)), 
                                            caption="Unsharp Mask (Weighted)", 
                                            use_container_width=True)
                            else:
                                st.error("Laplacian filtering failed.")
                
                elif operation == "Image Transformations":
                    st.header("🔄 Image Transformations")
                    st.markdown("Apply geometric transformations to images.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        scale_factor = st.slider(
                            "Scale Factor", 
                            0.1, 5.0, 2.0, 0.1,
                            help="Factor by which to scale the image"
                        )
                    with col2:
                        translate_x = st.slider(
                            "Translate X", 
                            -200, 200, 50,
                            help="Pixels to translate in X direction"
                        )
                        translate_y = st.slider(
                            "Translate Y", 
                            -200, 200, 50,
                            help="Pixels to translate in Y direction"
                        )
                    
                    if st.button("Apply Transformations", key="trans_btn"):
                        with st.spinner("Applying transformations..."):
                            scaled_img, swapped_img, rescaled_img, translated_img = atrans(
                                uploaded_file, scale_factor, translate_x, translate_y)
                            
                            if scaled_img is not None:
                                st.subheader("Transformation Results")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(scaled_img, 
                                            caption=f"Scaled (×{scale_factor})", 
                                            use_container_width=True)
                                    st.image(rescaled_img, 
                                            caption="Rescaled to Original Size", 
                                            use_container_width=True)
                                with col2:
                                    st.image(swapped_img, 
                                            caption="Left-Right Swapped", 
                                            use_container_width=True)
                                    st.image(translated_img, 
                                            caption=f"Translated ({translate_x}, {translate_y})", 
                                            use_container_width=True)
                            else:
                                st.error("Image transformations failed.")
                
                elif operation == "Walsh Transform":
                    st.header("🔳 Walsh Transform")
                    st.markdown("Generate Walsh matrices sorted by sequency and visualize basis images.")
                    
                    N = st.slider(
                        "Matrix Order (N)", 
                        2, 64, 8, 2,
                        help="Order of Walsh matrix (must be power of 2)"
                    )
                    
                    if st.button("Compute Walsh Matrix", key="walsh_btn"):
                        with st.spinner(f"Computing Walsh matrix of order {N}..."):
                            W_sorted = awalsh(N)
                            
                            if W_sorted is not None:
                                st.subheader(f"Walsh Matrix (Order {N}, Sorted by Sequency)")
                                st.dataframe(W_sorted)
                                
                                st.subheader("Walsh Basis Images")
                                fig, axes = plt.subplots(N, N, figsize=(10, 10))
                                for u in range(N):
                                    for v in range(N):
                                        basis_image = np.outer(W_sorted[u, :], W_sorted[v, :])
                                        axes[u, v].imshow(basis_image, cmap='gray')
                                        axes[u, v].axis('off')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Show application on image
                                st.subheader("Walsh Transform Application")
                                st.markdown("""
                                The Walsh transform is used in image processing and compression.
                                Below is an example of applying the transform to image blocks.
                                """)
                                
                                with st.spinner("Applying Walsh transform to image..."):
                                    img_gray = np.array(img.convert('L'), dtype=np.float64)
                                    block_size = 8
                                    h, w = img_gray.shape
                                    
                                    # Pad image to multiple of block size
                                    pad_h = (block_size - h % block_size) % block_size
                                    pad_w = (block_size - w % block_size) % block_size
                                    img_padded = np.pad(img_gray, ((0, pad_h), (0, pad_w)), mode='reflect')
                                    
                                    # Apply Walsh transform to each block
                                    transformed = np.zeros_like(img_padded)
                                    for i in range(0, img_padded.shape[0], block_size):
                                        for j in range(0, img_padded.shape[1], block_size):
                                            block = img_padded[i:i+block_size, j:j+block_size]
                                            transformed_block = np.dot(np.dot(W_sorted, block), W_sorted.T)
                                            transformed[i:i+block_size, j:j+block_size] = transformed_block
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.image(img_gray, 
                                                caption="Original Image", 
                                                use_container_width=True, 
                                                clamp=True)
                                    with col2:
                                        st.image(np.abs(transformed), 
                                                caption="Walsh Transform", 
                                                use_container_width=True, 
                                                clamp=True)
                            else:
                                st.error("Walsh matrix computation failed.")
                
                # Add a footer
                st.markdown("---")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        # Show welcome message and instructions when no image is uploaded
        st.info("ℹ️ Please upload an image to get started.")
        st.markdown("""
        ### How to use this toolbox:
        1. **Upload an image** using the file uploader above
        2. **Select an operation** from the sidebar
        3. **Adjust parameters** as needed
        4. **Click the apply button** to see results
        
        ### Supported operations:
        - **Contrast Stretching**: Adjust image contrast
        - **Convolution**: Perform 2D convolution
        - **DCT/DFT**: Frequency domain transforms
        - **Filtering**: Noise reduction techniques
        - **Histogram Equalization**: Improve contrast
        - **Edge Enhancement**: Laplacian filters
        - **Geometric Transformations**: Scaling, translation
        - **Hadamard/Walsh**: Orthogonal transforms
        """)

if __name__ == "__main__":
    main()