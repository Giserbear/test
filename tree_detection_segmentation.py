#!/usr/bin/env python3
"""
Tree Detection and Individual Tree Segmentation Script

This script implements the method described in the paper for:
1. Detecting treetops using local maximum filter
2. Individual tree segmentation based on regional growing method

Author: AI Assistant
Date: 2024
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima
from skimage.morphology import disk
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.features import shapes
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class TreeDetectionSegmentation:
    """
    A class for detecting treetops and segmenting individual trees from CHM data.
    """
    
    def __init__(self, chm_data: np.ndarray, pixel_size: float = 0.2):
        """
        Initialize the TreeDetectionSegmentation class.
        
        Args:
            chm_data: Canopy Height Model (CHM) data as 2D numpy array
            pixel_size: Size of each pixel in meters (default: 0.2m)
        """
        self.chm_data = chm_data.astype(np.float32)
        self.pixel_size = pixel_size
        self.height, self.width = chm_data.shape
        self.treetops = []
        self.segmented_regions = None
        
    def gaussian_median_filter(self, g: int = 5) -> np.ndarray:
        """
        Apply Gaussian median filter for noise elimination and low vegetation screening.
        
        Args:
            g: Window size of the Gaussian median filter
            
        Returns:
            Filtered CHM data
        """
        print(f"Applying Gaussian median filter with window size {g}...")
        
        # First apply Gaussian filter for smoothing
        gaussian_filtered = gaussian_filter(self.chm_data, sigma=g/3)
        
        # Then apply median filter for noise reduction
        median_filtered = median_filter(gaussian_filtered, size=g)
        
        # Screen out low vegetation below 3m
        filtered_chm = np.where(median_filtered < 3.0, 0, median_filtered)
        
        return filtered_chm
    
    def detect_treetops_local_maximum(self, filtered_chm: np.ndarray, 
                                    c: int = 15, min_height: float = 3.0) -> List[Tuple[int, int]]:
        """
        Detect treetops using circular moving window filter to find local maxima.
        
        Args:
            filtered_chm: Filtered CHM data
            c: Size of circular moving window filter
            min_height: Minimum height threshold for treetops
            
        Returns:
            List of treetop coordinates (row, col)
        """
        print(f"Detecting treetops using circular moving window of size {c}...")
        
        # Create circular kernel
        kernel = disk(c//2)
        
        # Apply maximum filter
        local_maxima = ndimage.maximum_filter(filtered_chm, footprint=kernel)
        
        # Find peaks where local maximum equals original value and height > min_height
        treetop_mask = (filtered_chm == local_maxima) & (filtered_chm > min_height)
        
        # Find coordinates of treetops
        treetop_coords = np.where(treetop_mask)
        treetops = list(zip(treetop_coords[0], treetop_coords[1]))
        
        # Remove treetops that are too close to each other (within 2*c pixels)
        filtered_treetops = self._filter_close_treetops(treetops, min_distance=2*c)
        
        print(f"Found {len(filtered_treetops)} treetops")
        self.treetops = filtered_treetops
        
        return filtered_treetops
    
    def _filter_close_treetops(self, treetops: List[Tuple[int, int]], 
                             min_distance: int) -> List[Tuple[int, int]]:
        """
        Remove treetops that are too close to each other.
        
        Args:
            treetops: List of treetop coordinates
            min_distance: Minimum distance between treetops in pixels
            
        Returns:
            Filtered list of treetop coordinates
        """
        if not treetops:
            return []
        
        # Sort by height (assuming we have access to CHM values)
        treetop_heights = [(self.chm_data[row, col], row, col) for row, col in treetops]
        treetop_heights.sort(reverse=True)
        
        filtered_treetops = []
        
        for height, row, col in treetop_heights:
            # Check if this treetop is far enough from already selected ones
            too_close = False
            for existing_row, existing_col in filtered_treetops:
                distance = np.sqrt((row - existing_row)**2 + (col - existing_col)**2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered_treetops.append((row, col))
        
        return filtered_treetops
    
    def region_growing_segmentation(self, filtered_chm: np.ndarray, 
                                  treetops: List[Tuple[int, int]]) -> np.ndarray:
        """
        Perform individual tree segmentation using region growing method.
        
        Args:
            filtered_chm: Filtered CHM data
            treetops: List of treetop coordinates as seed points
            
        Returns:
            Segmented regions as labeled array
        """
        print("Performing region growing segmentation...")
        
        # Initialize segmentation result
        segmented = np.zeros_like(filtered_chm, dtype=np.int32)
        region_id = 1
        
        # Process each treetop as a seed point
        for seed_row, seed_col in treetops:
            if segmented[seed_row, seed_col] != 0:
                continue  # Already processed
                
            # Get seed point height
            seed_height = filtered_chm[seed_row, seed_col]
            
            if seed_height < 3.0:  # Condition 3: seed point must exceed 3m
                continue
            
            # Grow region from this seed point
            region_pixels = self._grow_region(filtered_chm, seed_row, seed_col, 
                                            seed_height, segmented)
            
            if len(region_pixels) > 1:  # Only keep regions with more than 1 pixel
                for row, col in region_pixels:
                    segmented[row, col] = region_id
                region_id += 1
        
        self.segmented_regions = segmented
        print(f"Segmented into {region_id - 1} individual tree regions")
        
        return segmented
    
    def _grow_region(self, chm_data: np.ndarray, seed_row: int, seed_col: int, 
                    seed_height: float, segmented: np.ndarray) -> List[Tuple[int, int]]:
        """
        Grow a region from a seed point based on the specified conditions.
        
        Args:
            chm_data: CHM data
            seed_row, seed_col: Seed point coordinates
            seed_height: Height of the seed point
            segmented: Current segmentation result
            
        Returns:
            List of pixel coordinates belonging to the region
        """
        # Initialize region with seed point
        region_pixels = [(seed_row, seed_col)]
        to_check = [(seed_row, seed_col)]
        visited = set()
        
        while to_check:
            current_row, current_col = to_check.pop(0)
            
            if (current_row, current_col) in visited:
                continue
            visited.add((current_row, current_col))
            
            # Check 8-connected neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    neighbor_row = current_row + dr
                    neighbor_col = current_col + dc
                    
                    # Check bounds
                    if (neighbor_row < 0 or neighbor_row >= self.height or 
                        neighbor_col < 0 or neighbor_col >= self.width):
                        continue
                    
                    # Check if already processed
                    if segmented[neighbor_row, neighbor_col] != 0:
                        continue
                    
                    if (neighbor_row, neighbor_col) in visited:
                        continue
                    
                    neighbor_height = chm_data[neighbor_row, neighbor_col]
                    
                    # Apply growth conditions
                    if self._meets_growth_conditions(neighbor_height, seed_height, 
                                                   region_pixels, chm_data, 
                                                   seed_row, seed_col):
                        region_pixels.append((neighbor_row, neighbor_col))
                        to_check.append((neighbor_row, neighbor_col))
        
        return region_pixels
    
    def _meets_growth_conditions(self, pixel_height: float, seed_height: float,
                               region_pixels: List[Tuple[int, int]], chm_data: np.ndarray,
                               seed_row: int, seed_col: int) -> bool:
        """
        Check if a pixel meets the region growing conditions.
        
        Args:
            pixel_height: Height of the pixel to check
            seed_height: Height of the seed point
            region_pixels: Current region pixels
            chm_data: CHM data
            seed_row, seed_col: Seed point coordinates
            
        Returns:
            True if pixel meets all conditions
        """
        # Condition 1: pixel value must exceed 70% of seed point value
        if pixel_height < 0.7 * seed_height:
            return False
        
        # Condition 2: seed point must exceed 55% of average region value
        if len(region_pixels) > 0:
            region_heights = [chm_data[row, col] for row, col in region_pixels]
            avg_region_height = np.mean(region_heights)
            if seed_height < 0.55 * avg_region_height:
                return False
        
        # Condition 3: seed point must exceed 3m (already checked in main function)
        
        # Condition 4: maximum distance from seed point should be at most 50 pixels (10m)
        if len(region_pixels) > 0:
            max_distance = 0
            for row, col in region_pixels:
                distance = np.sqrt((row - seed_row)**2 + (col - seed_col)**2)
                max_distance = max(max_distance, distance)
            
            if max_distance >= 50:  # 50 pixels = 10m (assuming 0.2m pixel size)
                return False
        
        return True
    
    def regions_to_shapefile(self, output_path: str, 
                           transform: Optional[Tuple] = None) -> None:
        """
        Convert segmented regions to shapefile format.
        
        Args:
            output_path: Path to save the shapefile
            transform: Geospatial transform (if available)
        """
        print(f"Converting regions to shapefile: {output_path}")
        
        if self.segmented_regions is None:
            raise ValueError("No segmented regions available. Run segmentation first.")
        
        # Create geometries for each region
        geometries = []
        region_ids = []
        
        for region_id in np.unique(self.segmented_regions):
            if region_id == 0:  # Skip background
                continue
            
            # Get region mask
            region_mask = (self.segmented_regions == region_id)
            
            # Find contours
            contours, _ = cv2.findContours(
                region_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                if len(contour) >= 3:  # Need at least 3 points for a polygon
                    # Convert contour to polygon
                    polygon_coords = contour.reshape(-1, 2)
                    
                    # Convert pixel coordinates to geographic coordinates if transform is provided
                    if transform is not None:
                        # This is a simplified transformation - in practice you'd use rasterio.transform
                        polygon_coords = polygon_coords * self.pixel_size
                    
                    # Create polygon
                    polygon = Polygon(polygon_coords)
                    if polygon.is_valid and polygon.area > 0:
                        geometries.append(polygon)
                        region_ids.append(region_id)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'region_id': region_ids,
            'geometry': geometries
        })
        
        # Set CRS if transform is provided
        if transform is not None:
            gdf.crs = "EPSG:4326"  # Default CRS - adjust as needed
        
        # Save to shapefile
        gdf.to_file(output_path)
        print(f"Saved {len(geometries)} tree regions to {output_path}")
    
    def visualize_results(self, filtered_chm: np.ndarray, 
                         treetops: List[Tuple[int, int]], 
                         segmented: np.ndarray) -> None:
        """
        Visualize the detection and segmentation results.
        
        Args:
            filtered_chm: Filtered CHM data
            treetops: List of treetop coordinates
            segmented: Segmented regions
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original CHM
        axes[0].imshow(self.chm_data, cmap='viridis')
        axes[0].set_title('Original CHM')
        axes[0].axis('off')
        
        # Filtered CHM with treetops
        axes[1].imshow(filtered_chm, cmap='viridis')
        if treetops:
            treetop_rows, treetop_cols = zip(*treetops)
            axes[1].scatter(treetop_cols, treetop_rows, c='red', s=20, marker='x')
        axes[1].set_title('Filtered CHM with Treetops')
        axes[1].axis('off')
        
        # Segmented regions
        axes[2].imshow(segmented, cmap='tab20')
        axes[2].set_title('Segmented Individual Trees')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function demonstrating the tree detection and segmentation process.
    """
    # Create sample CHM data for demonstration
    print("Creating sample CHM data...")
    height, width = 200, 200
    
    # Generate synthetic CHM with some tree-like structures
    chm_data = np.zeros((height, width))
    
    # Add some tree-like structures
    tree_centers = [(50, 50), (100, 80), (150, 120), (80, 150), (120, 30)]
    for center_row, center_col in tree_centers:
        # Create circular tree canopy
        y, x = np.ogrid[:height, :width]
        mask = (x - center_col)**2 + (y - center_row)**2 <= 15**2
        tree_height = np.random.uniform(8, 15)
        chm_data[mask] = np.maximum(chm_data[mask], 
                                   tree_height * np.exp(-((x - center_col)**2 + (y - center_row)**2) / (2 * 8**2)))
    
    # Add some noise
    noise = np.random.normal(0, 0.5, (height, width))
    chm_data = np.maximum(0, chm_data + noise)
    
    # Initialize the tree detection and segmentation class
    tree_detector = TreeDetectionSegmentation(chm_data, pixel_size=0.2)
    
    # Step 1: Apply Gaussian median filter
    filtered_chm = tree_detector.gaussian_median_filter(g=5)
    
    # Step 2: Detect treetops using local maximum filter
    treetops = tree_detector.detect_treetops_local_maximum(filtered_chm, c=15)
    
    # Step 3: Perform region growing segmentation
    segmented = tree_detector.region_growing_segmentation(filtered_chm, treetops)
    
    # Step 4: Convert to shapefile
    tree_detector.regions_to_shapefile('individual_trees.shp')
    
    # Step 5: Visualize results
    tree_detector.visualize_results(filtered_chm, treetops, segmented)
    
    print("Tree detection and segmentation completed successfully!")


if __name__ == "__main__":
    main()