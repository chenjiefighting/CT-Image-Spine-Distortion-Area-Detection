# CT-Image-Spine-Distortion-Area-Detection
### Implementation of CT Image Spine Distortion Area Detection<br>
### Dataset:MICCAI 2018 IVDM3Seg Challenge<br>
The challenge dataset contains 16 3D multi-modal magnetic resonance (MR) scans of the lower spine, with their corresponding manual segmentations, collected from 8 subjects at two different stages in a study investigating intervertebral discs (IVD) degeneration.
###  Requires:
 nibabel<br>
 imageio<br>
 tensorflow 1.0<br>
 keras<br>
 hdf5<br>
 PIL<br>
 configparser<br>
 CV2<br>
###   Instructions:
 The format of the original data is .nii file<br>
 Convert original data into images, you can observe them intuitively<br>
 ``` python
 >>> nii_to_image.py 
 ```
 
 Convert image data into .hdf5 file<br>
 ``` python
 >>> rewrite_datasets.py
 ```
