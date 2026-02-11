Version status: v1.0

# GUV Filling Analysis – Circle Detection with OpenCV

This repository contains a custom Python script used to analyze **giant unilamellar vesicle (GUV)** filling dynamics over time.  
The analysis is based on **circle detection using OpenCV** and quantifies how much each GUV is filled relative to the surrounding background intensity.

---

## Overview

The goal of this analysis is to determine the **filling percentage** of GUVs at different time points from microscopy images.  
Each GUV is detected automatically using a circle detection algorithm, and its mean internal intensity is compared to the local background.

---

## Methodology

1. **Circle Detection**  
   - GUVs are identified using OpenCV’s Hough Circle Transform or a similar custom detection routine.
   - Detected circles define regions of interest (ROIs) corresponding to individual vesicles.

2. **Intensity Evaluation**  
   - For each detected GUV, the mean pixel intensity inside the circle is measured.
   - The filling percentage is calculated as:

     \[
     \text{Filling} = \frac{I_{\text{GUV}}}{I_{\text{background}}} \times 100
     \]

     where  
     - \( I_{\text{GUV}} \) = average intensity inside the vesicle  
     - \( I_{\text{background}} \) = average intensity of the background region

   - A **completely black vesicle** corresponds to **0% filling**,  
     while a **vesicle matching the average background intensity** corresponds to **100% filling**.

3. **Aggregation**  
   - The script computes the mean filling value for **all detected GUVs** across all tiles belonging to one sample at a specific time point.  
   - The result is a time series showing the evolution of average GUV filling.

---

## Output

- A .json table containing the filling percentage per GUV + average GUV filling per time point.
- Summary plots of mean filling vs. time can be generated for visualization.

---


How to run/install (Windows 11):

Option 1: Run dist/GUVprofiler.exe (no installation necessary, can be supplied on demand because it is >100MB)

Option 2: Manual installation:

    Install all required packages with pip install -r requirements.txt
    Run python main.py

How to use the script for batch analysis:

    Press add file and select files to be analyzed
    Specify all parameters such that they fit well to your experiment
    Press analyze
    After analysis is done, select timepoint on treeview, images of GUVs should appear:
      top left: raw image of GUVs
      top right: GUV binary mask
      lower left: green circles with detected GUVs
      lower right: red circles with accepted GUVs that are not stacked and have sufficiently low membrane intensity according to threshold
