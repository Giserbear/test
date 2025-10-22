# Tree Detection and Individual Tree Segmentation

This repository contains a Python implementation of tree detection and individual tree segmentation methods based on the paper's methodology. The script implements two main processes:

1. **Treetop Detection using Local Maximum Filter**
2. **Individual Tree Segmentation using Region Growing Method**

## Method Overview

### 3.2.2. Detecting Treetops using Local Maximum Filter

The method uses a two-step filtering process:

1. **Gaussian Median Filter**: Applies a bell-shaped Gaussian distribution filter with a narrow window size (equivalent to canopy size) for noise elimination in the CHM. Low vegetation below 3m is screened out.

2. **Circular Moving Window Filter**: Uses a circular moving window to calculate the majority value in a given range. The maximum value represents the tree peak, and the treetop position is obtained.

### 3.3. Individual Tree Segmentation based on Regional Growing Method

After detecting treetops, the method uses region growing algorithm to segment individual trees:

- **Seed Points**: Treetop locations from the local maximum method are used as initial seed points
- **Growth Conditions**:
  1. Pixel value in segmented region must exceed 70% of the seed point value
  2. Seed point value must exceed 55% of the average pixel value in the region
  3. Seed point value must exceed 3m
  4. Maximum distance between edge point and seed point is 50 pixels (10m)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from tree_detection_segmentation import TreeDetectionSegmentation
import numpy as np

# Load your CHM data (replace with your actual data loading)
chm_data = np.load('your_chm_data.npy')  # or load from raster file

# Initialize the detector
detector = TreeDetectionSegmentation(chm_data, pixel_size=0.2)

# Apply Gaussian median filter
filtered_chm = detector.gaussian_median_filter(g=5)

# Detect treetops
treetops = detector.detect_treetops_local_maximum(filtered_chm, c=15)

# Perform segmentation
segmented = detector.region_growing_segmentation(filtered_chm, treetops)

# Save results as shapefile
detector.regions_to_shapefile('individual_trees.shp')

# Visualize results
detector.visualize_results(filtered_chm, treetops, segmented)
```

### Running the Example

```bash
python tree_detection_segmentation.py
```

This will run the script with synthetic data and generate:
- Filtered CHM visualization
- Detected treetops
- Segmented individual trees
- Shapefile output (`individual_trees.shp`)

## Parameters

### Gaussian Median Filter
- `g`: Window size of the Gaussian median filter (default: 5)

### Local Maximum Detection
- `c`: Size of circular moving window filter (default: 15)
- `min_height`: Minimum height threshold for treetops (default: 3.0m)

### Region Growing
- Growth conditions are hardcoded as per the paper:
  - 70% threshold for pixel inclusion
  - 55% threshold for seed point validation
  - 3m minimum height
  - 50 pixel maximum distance (10m)

## Output

The script generates:
1. **Filtered CHM**: Noise-reduced canopy height model
2. **Treetop Coordinates**: List of detected treetop positions
3. **Segmented Regions**: Labeled array with individual tree regions
4. **Shapefile**: Vector format of individual tree polygons

## File Structure

```
├── tree_detection_segmentation.py  # Main implementation
├── requirements.txt                # Python dependencies
├── README.md                      # This file
└── individual_trees.shp           # Output shapefile (generated)
```

## Dependencies

- numpy: Numerical computations
- opencv-python: Image processing
- scipy: Scientific computing
- scikit-image: Image processing algorithms
- geopandas: Geospatial data handling
- shapely: Geometric operations
- rasterio: Raster data I/O
- matplotlib: Visualization

## Notes

- The script assumes CHM data is in meters
- Pixel size is configurable (default: 0.2m)
- The method works best with high-resolution CHM data
- Parameters may need adjustment based on your specific data characteristics

## Citation

If you use this code, please cite the original paper that describes this methodology.