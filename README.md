# CT-Image-Spine-Distortion-Area-Detection
 Implementation of CT Image Spine Distortion Area Detection<br>
 ==
 Dataset:MICCAI 2018 IVDM3Seg Challenge<br>
 ==
 
 requires:<br>
 nibabel<br>
 imageio<br>
 tensorflow 1.0<br>
 keras<br>
 hdf5<br>
 PIL<br>
 configparser<br>
 CV2<br>
 
 The format of the original data is .nii file<br>
 Convert original data into images, you can observe them intuitively<br>
 $run nii_to_image.py <br>
 
 Convert image data into .hdf5 file
 $run rewrite_datasets.py
 
