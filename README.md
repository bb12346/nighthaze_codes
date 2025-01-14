# NightHaze: Nighttime Image Dehazing via Self-Prior Learning (AAAI'25)

<div align="center">
  
[![demo platform](https://img.shields.io/badge/NightHaze%20project%20page-lightblue)](https://bb12346.github.io/Proj_NightHaze/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2403.07408-b31b1b.svg)](https://arxiv.org/abs/2403.07408)&nbsp;

</div>

## Abstract
Masked autoencoder (MAE) shows that severe augmentation during training produces robust representations for high-level tasks. This paper brings the MAE-like framework to nighttime image enhancement, demonstrating that severe augmentation during training produces strong network priors that are resilient to real-world night haze degradations. We propose a novel nighttime image dehazing method with self-prior learning. Our main novelty lies in the design of severe augmentation, which allows our model to learn robust priors. Unlike MAE that uses masking, we leverage two key challenging factors of nighttime images as augmentation: light effects and noise. During training, we intentionally degrade clear images by blending them with light effects as well as by adding noise, and subsequently restore the clear images. This enables our model to learn clear background priors. By increasing the noise values to approach as high as the pixel intensity values of the glow and light effect blended images, our augmentation becomes severe, resulting in stronger priors. While our self-prior learning is considerably effective in suppressing glow and revealing details of background scenes, in some cases, there are still some undesired artifacts that remain, particularly in the forms of over-suppression. To address these artifacts, we propose a self-refinement module based on the semi-supervised teacher-student framework. Our NightHaze, especially our MAE-like self-prior learning, shows that models trained with severe augmentation effectively improve the visibility of input haze images, approaching the clarity of clear nighttime images. Extensive experiments demonstrate that our NightHaze achieves state-of-the-art performance, outperforming existing nighttime image dehazing methods by a substantial margin of 15.5% for MUSIQ and 23.5% for ClipIQA.

## Inference Scripts

To test our Nighthaze, you can run the following command:
```shell
1. Change data_dir: "" in RealSHaze.yml to the desired dataset path.
For example: "/home/Beibei/NightHaze/data/"

2. Add your test folder to self.trainlist and self.testlist in datasets/realshaze_add.py.
For example:
self.trainlist = ['haze']
self.testlist = ['haze']

3. Run the following command using --sid "haze":
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u eval_diffusion.py \
    --sid "haze" \
    --config "RealSHaze.yml" \
    --resume 'param/NightHaze_SPL_2w.pth.tar' > log_test.txt 2>&1 &

```

## Acknowledgments
Code is implemented based [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion), we would like to thank them.


### Citation
If this work is useful for your research, please cite our paper. 
```BibTeX
@article{lin2024nighthaze,
  title={NightHaze: Nighttime Image Dehazing via Self-Prior Learning},
  author={Lin, Beibei and Jin, Yeying and Yan, Wending and Ye, Wei and Yuan, Yuan and Tan, Robby T},
  journal={arXiv preprint arXiv:2403.07408},
  year={2024}
}
```
