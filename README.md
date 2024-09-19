<h2>Tensorflow-Image-Segmentation-Modified-Bone-Marrow-Cell (2024/09/19)</h2>

This is the first experiment of Image Segmentation for Modified-Bone-Marrow (MBM) Cell based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
Pre-Augmented 640x640 pixels
<a href="https://drive.google.com/file/d/1xsEr_evZ_9sLr-TGyAnkEkPt-lLBLMLa/view?usp=sharing">
MBM-ImageMask-Dataset.zip</a>, which was derived by us from 
<a href="https://github.com/ieee8023/countception/blob/master/MBM_data.zip">
MBM_data.zip</a> for counting objects.
<br>
<br>
<!--
Please see also 
<a href="https://github.com/sarah-antillia/ImageMask-Dataset-Modified-Bone-Marrow">
ImageMask-Dataset-Modified-Bone-Marrow</a><br>
-->

<hr>
<b>Actual Image Segmentation for Images of 640x640 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/images/barrdistorted_101_0.3_0.3_1019.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/masks/barrdistorted_101_0.3_0.3_1019.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test_output/barrdistorted_101_0.3_0.3_1019.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/images/barrdistorted_101_0.3_0.3_1029.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/masks/barrdistorted_101_0.3_0.3_1029.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test_output/barrdistorted_101_0.3_0.3_1029.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/images/barrdistorted_105_0.3_0.3_1044.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/masks/barrdistorted_105_0.3_0.3_1044.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test_output/barrdistorted_105_0.3_0.3_1044.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this MBM Segmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1 Dataset Citation</h3>
We used <a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a>
to create the Pre-Augmented segmentation dataset from a point-wise dataset for objects-counting
<a href="https://github.com/ieee8023/countception/blob/master/MBM_data.zip">
MBM_data.zip</a><br>
 in <a href="https://github.com/ieee8023/countception">https://github.com/ieee8023/countception
</a>.<br><br>

<b>Citation request:</b><br>
Count-ception: Counting by Fully Convolutional Redundant Counting<br>
JP Cohen, G Boucher, CA Glastonbury, HZ Lo, Y Bengio<br>
International Conference on Computer Vision (ICCV) Workshop on Bioimage Computing<br>

@inproceedings{Cohen2017,<br>
title = {Count-ception: Counting by Fully Convolutional Redundant Counting},<br>
author = {Cohen, Joseph Paul and Boucher, Genevieve and Glastonbury, Craig A. and Lo, Henry Z. and Bengio, Yoshua},<br>
booktitle = {International Conference on Computer Vision Workshop on BioImage Computing},
url = {http://arxiv.org/abs/1703.08710},<br>
year = {2017}<br>
}<br>
<br>
<h3>2 MBM ImageMask Dataset
</h3>
 If you would like to train this MBM Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1xsEr_evZ_9sLr-TGyAnkEkPt-lLBLMLa/view?usp=sharing">
MBM-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─MBM
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>

<b>MBM Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/MBM/MBM-ImageMask-Dataset_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough large to use for a training set for our segmentation model.
 <br>

<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>3 Train TensorflowUNet Model</h3>
 We have trained MBMTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/MBM/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/MBM and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>


<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
model          = "TensorflowUNet"
image_width    = 640
image_height   = 640
image_channels = 3
num_classes    = 1
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 7
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00008
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 1
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for an image in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>
<br>

In this experiment, the training process was stopped at epoch 99 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/asset/train_console_output_at_epoch_99.png" width="720" height="auto"><br>
<br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/MBM/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/MBM/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/MBM</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for MBM.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/asset/evaluate_console_output_at_epoch_99.png" width="720" height="auto">
<br><br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/MBM/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this MBM/test was not so low, and dice_coef not so high as shown below.
<br>
<pre>
loss,0.2146
dice_coef,0.7351
</pre>


<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/MBM</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for MBM.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/images/barrdistorted_102_0.3_0.3_1008.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/masks/barrdistorted_102_0.3_0.3_1008.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test_output/barrdistorted_102_0.3_0.3_1008.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/images/barrdistorted_103_0.3_0.3_1006.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/masks/barrdistorted_103_0.3_0.3_1006.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test_output/barrdistorted_103_0.3_0.3_1006.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/images/barrdistorted_105_0.3_0.3_1044.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/masks/barrdistorted_105_0.3_0.3_1044.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test_output/barrdistorted_105_0.3_0.3_1044.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/images/distorted_0.02_rsigma0.5_sigma40_1007.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/masks/distorted_0.02_rsigma0.5_sigma40_1007.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test_output/distorted_0.02_rsigma0.5_sigma40_1007.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/images/pincdistorted_102_0.3_-0.3_1016.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test/masks/pincdistorted_102_0.3_-0.3_1016.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MBM/mini_test_output/pincdistorted_102_0.3_-0.3_1016.jpg" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Count-Ception: Counting by Fully Convolutional Redundant Counting (arXiv)</b><br>
Joseph Paul Cohen, Genevieve Boucher, Craig A. Glastonbury, Henry Z. Lo, Yoshua Bengio<br>
<a href="https://arxiv.org/pdf/1703.08710">https://arxiv.org/pdf/1703.08710</a>
<br>
<br>
<b>2. SAU-Net: A Universal Deep Network for Cell Counting</b><br>
Yue Guo, Guorong Wu, Jason Stein, and Ashok Krishnamurthy<br>
ACM BCB. 2019 Sep; 2019: 299–306.<br>
doi: 10.1145/3307339.3342153<br>
<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8153189/">https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8153189/</a>
<br><br>
<b>3. Machine-Based Detection and Classification for Bone Marrow Aspirate Differential Counts: <br>
Initial Development Focusing on Non-Neoplastic Cells
</b><br>
 doi: 10.1038/s41374-019-0325-7<br>
<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6920560/">https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6920560/</a>

<br>
<br>
<b>4. MSCA-UNet: Multi-Scale Convolutional Attention UNet for Automatic Cell Counting Using Density Regression</b><br>
L. Qian, W. Qian, D. Tian, Y. Zhu, H. Zhao and Y. Yao, <br> 
doi: 10.1109/ACCESS.2023.3304993.<br>
<a href="https://ieeexplore.ieee.org/document/10216972">https://ieeexplore.ieee.org/document/10216972</a>
<br>
<br>

