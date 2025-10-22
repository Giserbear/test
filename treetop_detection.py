#!/usr/bin/env python3
"""
Treetop Detection and Individual Tree Segmentation Script

This script implements the methods described in the paper for:
1. Detecting treetops using local maximum filter with Gaussian median filtering
2. Individual tree segmentation based on region growing method

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
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.features import shapes
import warnings
warnings.filterwarnings('ignore')


class TreetopDetector:
    """
    Class for detecting treetops using local maximum filter approach
    """
    
    def __init__(self, gaussian_window_size=5, circular_window_size=7):
        """
        Initialize the treetop detector
        
        Args:
            gaussian_window_size (int): Window size for Gaussian median filter
            circular_window_size (int): Size of circular moving window filter
        """
        self.gaussian_window_size = gaussian_window_size
        self.circular_window_size = circular_window_size
        
    def preprocess_chm(self, chm, min_height=3.0):
        """
        Preprocess CHM by applying Gaussian median filter and height thresholding
        
        Args:
            chm (numpy.ndarray): Canopy Height Model
            min_height (float): Minimum height threshold (default: 3m)
            
        Returns:
            numpy.ndarray: Preprocessed CHM
        """
        print("Preprocessing CHM...")
        
        # Apply Gaussian median filter for noise elimination
        # Using a combination of Gaussian and median filters
        gaussian_filtered = gaussian_filter(chm, sigma=self.gaussian_window_size/3)
        median_filtered = median_filter(gaussian_filtered, size=self.gaussian_window_size)
        
        # Screen out low vegetation below threshold
        processed_chm = np.where(median_filtered >= min_height, median_filtered, 0)
        
        return processed_chm
    
    def detect_local_maxima(self, chm):
        """
        Detect local maxima using circular moving window filter
        
        Args:
            chm (numpy.ndarray): Preprocessed CHM
            
        Returns:
            tuple: (treetop_coords, treetop_heights)
        """
        print("Detecting local maxima...")
        
        # Create circular kernel for moving window
        kernel_size = self.circular_window_size
        kernel = disk(kernel_size // 2)
        
        # Apply maximum filter
        local_maxima = ndimage.maximum_filter(chm, footprint=kernel)
        
        # Find peaks where the original value equals the local maximum
        # and the value is greater than 0
        peaks = (chm == local_maxima) & (chm > 0)
        
        # Get coordinates and heights of treetops
        treetop_coords = np.where(peaks)
        treetop_heights = chm[peaks]
        
        # Sort by height (descending)
        sorted_indices = np.argsort(treetop_heights)[::-1]
        treetop_coords = (treetop_coords[0][sorted_indices], 
                         treetop_coords[1][sorted_indices])
        treetop_heights = treetop_heights[sorted_indices]
        
        print(f"Found {len(treetop_heights)} potential treetops")
        
        return treetop_coords, treetop_heights
    
    def detect_treetops(self, chm, min_height=3.0):
        """
        Complete treetop detection pipeline
        
        Args:
            chm (numpy.ndarray): Canopy Height Model
            min_height (float): Minimum height threshold
            
        Returns:
            tuple: (processed_chm, treetop_coords, treetop_heights)
        """
        # Preprocess CHM
        processed_chm = self.preprocess_chm(chm, min_height)
        
        # Detect local maxima
        treetop_coords, treetop_heights = self.detect_local_maxima(processed_chm)
        
        return processed_chm, treetop_coords, treetop_heights


class TreeSegmenter:
    """
    Class for individual tree segmentation using region growing method
    """
    
    def __init__(self, seed_height_ratio=0.7, region_height_ratio=0.55, 
                 min_seed_height=3.0, max_distance=50):
        """
        Initialize the tree segmenter
        
        Args:
            seed_height_ratio (float): Pixel value must exceed this ratio of seed point value
            region_height_ratio (float): Seed point must exceed this ratio of average region value
            min_seed_height (float): Minimum seed point height (meters)
            max_distance (int): Maximum distance from seed point (pixels)
        """
        self.seed_height_ratio = seed_height_ratio
        self.region_height_ratio = region_height_ratio
        self.min_seed_height = min_seed_height
        self.max_distance = max_distance
    
    def region_growing(self, chm, seed_points):
        """
        Perform region growing segmentation starting from seed points
        
        Args:
            chm (numpy.ndarray): Canopy Height Model
            seed_points (list): List of (row, col) seed point coordinates
            
        Returns:
            numpy.ndarray: Segmented regions (labeled array)
        """
        print("Performing region growing segmentation...")
        
        height, width = chm.shape
        segmented = np.zeros_like(chm, dtype=int)
        region_id = 1
        
        # 8-connectivity for neighborhood expansion
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for seed_row, seed_col in seed_points:
            if segmented[seed_row, seed_col] != 0:  # Already assigned to a region
                continue
                
            seed_height = chm[seed_row, seed_col]
            if seed_height < self.min_seed_height:
                continue
            
            # Initialize region
            region_pixels = [(seed_row, seed_col)]
            segmented[seed_row, seed_col] = region_id
            
            # Region growing
            queue = [(seed_row, seed_col)]
            
            while queue:
                current_row, current_col = queue.pop(0)
                
                # Check all 8 neighbors
                for dr, dc in directions:
                    new_row, new_col = current_row + dr, current_col + dc
                    
                    # Check bounds
                    if (0 <= new_row < height and 0 <= new_col < width and 
                        segmented[new_row, new_col] == 0):
                        
                        pixel_height = chm[new_row, new_col]
                        
                        # Check growth conditions
                        if self._meets_growth_conditions(pixel_height, seed_height, 
                                                       region_pixels, chm, 
                                                       new_row, new_col, seed_row, seed_col):
                            segmented[new_row, new_col] = region_id
                            region_pixels.append((new_row, new_col))
                            queue.append((new_row, new_col))
            
            region_id += 1
        
        print(f"Created {region_id - 1} segmented regions")
        return segmented
    
    def _meets_growth_conditions(self, pixel_height, seed_height, region_pixels, 
                                chm, pixel_row, pixel_col, seed_row, seed_col):
        """
        Check if a pixel meets the region growing conditions
        
        Args:
            pixel_height (float): Height of the candidate pixel
            seed_height (float): Height of the seed point
            region_pixels (list): List of pixels already in the region
            chm (numpy.ndarray): Canopy Height Model
            pixel_row, pixel_col (int): Coordinates of candidate pixel
            seed_row, seed_col (int): Coordinates of seed point
            
        Returns:
            bool: True if pixel meets growth conditions
        """
        # Condition 1: Pixel value must exceed 70% of seed point value
        if pixel_height < self.seed_height_ratio * seed_height:
            return False
        
        # Condition 2: Seed point must exceed 55% of average region value
        if len(region_pixels) > 0:
            region_heights = [chm[r, c] for r, c in region_pixels]
            avg_region_height = np.mean(region_heights)
            if seed_height < self.region_height_ratio * avg_region_height:
                return False
        
        # Condition 3: Seed point must exceed 3m (already checked in main function)
        
        # Condition 4: Maximum distance constraint
        distance = np.sqrt((pixel_row - seed_row)**2 + (pixel_col - seed_col)**2)
        if distance > self.max_distance:
            return False
        
        return True
    
    def segment_trees(self, chm, treetop_coords):
        """
        Complete tree segmentation pipeline
        
        Args:
            chm (numpy.ndarray): Canopy Height Model
            treetop_coords (tuple): Treetop coordinates from detection
            
        Returns:
            numpy.ndarray: Segmented regions
        """
        # Convert treetop coordinates to list of (row, col) tuples
        seed_points = list(zip(treetop_coords[0], treetop_coords[1]))
        
        # Perform region growing
        segmented_regions = self.region_growing(chm, seed_points)
        
        return segmented_regions


class TreeAnalysisPipeline:
    """
    Complete pipeline for treetop detection and tree segmentation
    """
    
    def __init__(self, gaussian_window=5, circular_window=7, 
                 seed_height_ratio=0.7, region_height_ratio=0.55,
                 min_height=3.0, max_distance=50):
        """
        Initialize the complete pipeline
        
        Args:
            gaussian_window (int): Gaussian median filter window size
            circular_window (int): Circular moving window size
            seed_height_ratio (float): Seed height ratio for region growing
            region_height_ratio (float): Region height ratio for region growing
            min_height (float): Minimum tree height threshold
            max_distance (int): Maximum distance for region growing
        """
        self.detector = TreetopDetector(gaussian_window, circular_window)
        self.segmenter = TreeSegmenter(seed_height_ratio, region_height_ratio, 
                                     min_height, max_distance)
    
    def process(self, chm, min_height=3.0):
        """
        Complete processing pipeline
        
        Args:
            chm (numpy.ndarray): Canopy Height Model
            min_height (float): Minimum tree height threshold
            
        Returns:
            dict: Results containing processed CHM, treetops, and segments
        """
        print("Starting tree analysis pipeline...")
        
        # Step 1: Detect treetops
        processed_chm, treetop_coords, treetop_heights = self.detector.detect_treetops(chm, min_height)
        
        # Step 2: Segment individual trees
        segmented_regions = self.segmenter.segment_trees(processed_chm, treetop_coords)
        
        results = {
            'processed_chm': processed_chm,
            'treetop_coords': treetop_coords,
            'treetop_heights': treetop_heights,
            'segmented_regions': segmented_regions,
            'num_trees': len(treetop_heights)
        }
        
        print(f"Pipeline completed. Found {results['num_trees']} trees.")
        
        return results
    
    def visualize_results(self, chm, results, save_path=None):
        """
        Visualize the analysis results
        
        Args:
            chm (numpy.ndarray): Original CHM
            results (dict): Results from processing pipeline
            save_path (str): Path to save the visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original CHM
        im1 = axes[0, 0].imshow(chm, cmap='viridis')
        axes[0, 0].set_title('Original CHM')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Processed CHM
        im2 = axes[0, 1].imshow(results['processed_chm'], cmap='viridis')
        axes[0, 1].set_title('Processed CHM (Filtered)')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Treetops
        axes[1, 0].imshow(results['processed_chm'], cmap='viridis', alpha=0.7)
        treetop_coords = results['treetop_coords']
        axes[1, 0].scatter(treetop_coords[1], treetop_coords[0], 
                          c='red', s=20, marker='x', linewidths=2)
        axes[1, 0].set_title(f'Treetops Detected ({len(treetop_coords[0])} trees)')
        axes[1, 0].axis('off')
        
        # Segmented regions
        im4 = axes[1, 1].imshow(results['segmented_regions'], cmap='tab20')
        axes[1, 1].set_title('Individual Tree Segments')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def export_to_shapefile(self, segmented_regions, chm, output_path, pixel_size=0.2):
        """
        Export segmented regions to shapefile
        
        Args:
            segmented_regions (numpy.ndarray): Segmented regions
            chm (numpy.ndarray): CHM for height information
            output_path (str): Output shapefile path
            pixel_size (float): Pixel size in meters
        """
        print("Exporting segments to shapefile...")
        
        # Convert raster to vector
        shapes_list = []
        properties_list = []
        
        for region_id in np.unique(segmented_regions):
            if region_id == 0:  # Skip background
                continue
                
            # Create mask for current region
            mask = (segmented_regions == region_id)
            
            # Get region properties
            region_heights = chm[mask]
            max_height = np.max(region_heights)
            mean_height = np.mean(region_heights)
            area_pixels = np.sum(mask)
            area_m2 = area_pixels * (pixel_size ** 2)
            
            # Convert to polygon
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) >= 3:  # Need at least 3 points for polygon
                    # Convert pixel coordinates to real coordinates
                    polygon_coords = []
                    for point in contour:
                        x = point[0][1] * pixel_size  # Convert row to y
                        y = point[0][0] * pixel_size  # Convert col to x
                        polygon_coords.append([x, y])
                    
                    if len(polygon_coords) >= 3:
                        polygon = Polygon(polygon_coords)
                        shapes_list.append(polygon)
                        properties_list.append({
                            'tree_id': int(region_id),
                            'max_height': float(max_height),
                            'mean_height': float(mean_height),
                            'area_m2': float(area_m2),
                            'area_pixels': int(area_pixels)
                        })
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(properties_list, geometry=shapes_list)
        gdf.crs = "EPSG:4326"  # Set appropriate CRS
        
        # Save to shapefile
        gdf.to_file(output_path)
        print(f"Shapefile saved to {output_path}")


def create_sample_chm(size=(200, 200), num_trees=10, noise_level=0.1):
    """
    Create a sample CHM for testing purposes
    
    Args:
        size (tuple): Size of the CHM (height, width)
        num_trees (int): Number of trees to simulate
        noise_level (float): Level of noise to add
        
    Returns:
        numpy.ndarray: Sample CHM
    """
    height, width = size
    chm = np.zeros((height, width))
    
    # Add some random trees
    np.random.seed(42)
    for _ in range(num_trees):
        # Random tree position
        center_row = np.random.randint(20, height - 20)
        center_col = np.random.randint(20, width - 20)
        
        # Random tree parameters
        tree_height = np.random.uniform(5, 25)
        tree_radius = np.random.uniform(5, 15)
        
        # Create circular tree canopy
        y, x = np.ogrid[:height, :width]
        mask = (x - center_col)**2 + (y - center_row)**2 <= tree_radius**2
        
        # Add height with Gaussian distribution
        for r in range(height):
            for c in range(width):
                if mask[r, c]:
                    distance = np.sqrt((r - center_row)**2 + (c - center_col)**2)
                    if distance <= tree_radius:
                        # Gaussian height distribution
                        height_factor = np.exp(-(distance**2) / (2 * (tree_radius/3)**2))
                        chm[r, c] = max(chm[r, c], tree_height * height_factor)
    
    # Add noise
    noise = np.random.normal(0, noise_level, chm.shape)
    chm = np.maximum(0, chm + noise)
    
    return chm


def main():
    """
    Main function demonstrating the tree analysis pipeline
    """
    print("Tree Analysis Pipeline Demo")
    print("=" * 50)
    
    # Create sample CHM
    print("Creating sample CHM...")
    chm = create_sample_chm(size=(150, 150), num_trees=8)
    
    # Initialize pipeline with parameters from Table 2 (example values)
    pipeline = TreeAnalysisPipeline(
        gaussian_window=5,      # g parameter
        circular_window=7,      # c parameter
        seed_height_ratio=0.7,  # 70% of seed point value
        region_height_ratio=0.55, # 55% of average region value
        min_height=3.0,         # 3m minimum height
        max_distance=50         # 50 pixels maximum distance
    )
    
    # Process the CHM
    results = pipeline.process(chm, min_height=3.0)
    
    # Print results
    print(f"\nResults:")
    print(f"Number of trees detected: {results['num_trees']}")
    print(f"Treetop heights range: {np.min(results['treetop_heights']):.2f} - {np.max(results['treetop_heights']):.2f} m")
    
    # Visualize results
    pipeline.visualize_results(chm, results, save_path='tree_analysis_results.png')
    
    # Export to shapefile
    pipeline.export_to_shapefile(
        results['segmented_regions'], 
        results['processed_chm'], 
        'individual_trees.shp',
        pixel_size=0.2
    )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()