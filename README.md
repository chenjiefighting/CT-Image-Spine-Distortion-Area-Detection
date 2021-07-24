# CT-Image-Spine-Distortion-Area-Detection
### Implementation of CT Image Spine Distortion Area Detection<br>
### Dataset:MICCAI 2018 IVDM3Seg Challenge<br>
The challenge dataset contains 16 3D multi-modal magnetic resonance (MR) scans of the lower spine, with their corresponding manual segmentations, collected from 8 subjects at two different stages in a study investigating intervertebral discs (IVD) degeneration. You can get by https://www.dropbox.com/s/q6eb6gjr0v5huwh/Training.zip?dl=0
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


###   Results:
![001_test](https://user-images.githubusercontent.com/76989858/118978198-a1cb5100-b9a9-11eb-9e13-ac9c7b446895.png)
![1](https://user-images.githubusercontent.com/76989858/118978253-b14a9a00-b9a9-11eb-9bcf-d859cbb89d87.png)<br>
explanation:blue represents correct segmentation, green represents missed segmentation, and red represents wrong segmentation<br>
|Indicators    |PA             |MPA           |MIoU          |
|------------- |----------:    |-------------:|:------------:|
|Value         |0.996          |0.896         |0.875         |
