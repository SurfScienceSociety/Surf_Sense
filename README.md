## Project description

This repository contains a set of desktop tools for end-to-end processing and visualization of instrumented surfboard data.  
The focus is on synchronizing multi-sensor logs (FFS, IMU and GNSS/GPS) with video, and providing an interactive environment for
surf biomechanics and performance analysis.

The main application, **SurfSense**, implements a processing pipeline that:
- preprocesses raw CSV/txt logs from the embedded system
- treat data, support video annotations and organize metadata
- builds a structured SQLite database from the cleaned data
- filters and interpolates sensor signals
- exposes selected events for detailed inspection

On top of this pipeline, several PyQt-based dashboards offer:
- time-series visualization of FFS, IMU and $GNRMC (speed/heading) variables
- GPS track and speed maps (with Cartopy-based trajectory rendering)
- synchronized video playback (e.g. GoPro footage) aligned with sensor timelines
- a simple 3D surfboard viewer driven by IMU yaw/pitch/roll

These tools are being developed primarily for research on surfboard dynamics, wave riding technique and board–water interaction.
The code is still a work in progress / experimental and may change frequently as the analysis pipeline evolves.
