# Q-RBSA
High-Resolution 3D EBSD Map Generation Using An Efficient Quaternion Transformer Network

[Devendra K. Jangid](https://sites.google.com/view/dkj910), [Neal R. Brodnik](https://scholar.google.com/citations?user=3dAoFJkAAAAJ&hl=en), [McLean P. Echlin](https://scholar.google.com/citations?user=fxN2OsUAAAAJ&hl=en), [Tresa M. Pollock](https://materials.ucsb.edu/people/faculty/tresa-pollock), [Samantha H. Daly](https://scholar.google.com/citations?user=3whYx4UAAAAJ&hl=en), [B.S. Manjunath](https://scholar.google.com/citations?user=wRYM4qgAAAAJ&hl=en)

[Paper](https://arxiv.org/abs/2303.10722)

<hr />

> **Abstract:** *Gathering 3D material microstructural information is time-consuming, expensive, and energy-intensive. Acquisition of 3D data has been accelerated by developments in serial sectioning instrument capabilities; however, for crystallographic information, the electron backscatter diffraction (EBSD) imaging modality remains rate limiting. We propose a physics-based efficient deep learning framework to reduce the time and cost of collecting 3D EBSD maps. Our framework uses a quaternion residual block self-attention network (QRBSA) to generate high-resolution 3D EBSD maps from sparsely sectioned EBSD maps. In QRBSA, quaternion-valued convolution effectively learns local relations in orientation space, while self-attention in the quaternion domain captures long-range correlations. We apply our framework to 3D data collected from commercially relevant titanium alloys, showing both qualitatively and quantitatively that our method can predict missing samples (EBSD information between sparsely sectioned mapping points) as compared to high-resolution ground truth 3D EBSD maps.*
<hr />


# Input: 

A Sparsely Sectioned 3D EBSD Sample in Quaternion Orientation (.npy array) 

# Output: 

        A High-Resolution 3D EBSD Sample in Quaternion Orientation (.npy array)

        A High-Resolution 3D EBSD Sample (Dream 3D format)
        
        A Sparsely Sectioned Input 3D Sample (Dream 3D format)


