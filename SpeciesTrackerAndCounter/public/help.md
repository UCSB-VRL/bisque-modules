# Species Tracker And Counter

This module is a demostration for the paper, Context-Driven Detection of Invertebrate Species in Deep-Sea Video [1], by McEver et al. It detects and tracks 10 deep-sea species trained by videos from the DUSIA dataset (Dataset for Underwater Substrate and Invertebrate Analysis). The DUSIA dataset currently includes over ten hours of footage across 25 videos captured in 1080p at 30 fps by an ROV following pre-planned transects across the ocean floor near the Channel Islands of California. The 10 species include basket star (BS), fragile pink urchin (FPU), gray gorgonian (GG), long-legged sunflower star (LLS), red swifita gorgonian (RSG), squat lobster (SL), laced sponge (LS), white slipper sea cucumber (WSSC), white spine sea cucumber (WSpSC), and yellow gorgonian (YG).

## Input
* DUSIA under water videos (*example.mp4*)

## Output
* Videos with detection, track, and counts labels (*example_output.mp4*)
* Species counts table (*example_count.hdf5*)
* Annotation file (*example_annotations.txt*)

## Reference
[1] McEver, R.A., Zhang, B., Levenson, C. et al. Context-Driven Detection of Invertebrate Species in Deep-Sea Video. Int J Comput Vis (2023). https://doi.org/10.1007/s11263-023-01755-4