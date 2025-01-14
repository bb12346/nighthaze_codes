# NightHaze: Nighttime Image Dehazing via Self-Prior Learning (AAAI'25)

<div align="center">
  
[![demo platform](https://img.shields.io/badge/NightHaze%20project%20page-lightblue)](https://bb12346.github.io/Proj_NightHaze/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2403.07408-b31b1b.svg)](https://arxiv.org/abs/2403.07408)&nbsp;

</div>


## Inference Scripts

To test our Nighthaze, you can run the following command:
```shell
1. change data_dir: "" in RealSHaze.yml to data_dir. 
e.g. "/home/Beibei/NightHaze/data/"

2. add your test folder to  self.trainlist and self.testlist in datasets/realshaze_add.py
e.g.
self.trainlist = ['haze']
self.testlist = ['haze']

3. run the following codes using --sid "haze" 
CUDA_VISIBLE_DEVICES=3,4,5,6,7 nohup python -u eval_diffusion.py --sid "haze" --config "RealSHaze.yml" --resume 'param/NightHaze_SPL_2w.pth.tar' > log_test.txt 2>&1 &


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
