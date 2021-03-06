# MSBGM
The code for the following paper in IEEE Access:
> [Perceived Image Decoding From Brain Activity Using Shared Information of Multi-Subject fMRI Data](https://ieeexplore.ieee.org/abstract/document/9349437)

![Image 1](Method.png)

# Code
- MSBGM-MLP.ipynb : code of the proposed method to reproduce TABLE I and FIGURE 6 (i) of the manuscript (Jupyter Notebook)
- MSBGM-MLP.py : code of the proposed method to reproduce TABLE I and FIGURE 6 (i) of the manuscript (Python3.7)
- MVBGM-MS.ipynb : code of the method in Ref. [17] to reproduce TABLE I and FIGURE 6 (i) of the manuscript (Jupyter Notebook)
- MVBGM-MS.py : code of the method in Ref. [17] to reproduce TABLE I and FIGURE 6 (i) of the manuscript (Python3.7)

Note that the above codes do not reproduce exactly the same results as TABLE I and FIGURE 6 (i) since prior distributions are randomly initialized by a multivariate normal distribution.

[17] Y. Akamatsu, R. Harakawa, T. Ogawa, and M. Haseyama, "Multi-view Bayesian generative model for multi-subject fMRI data on brain decoding of viewed image categories," in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), 2020, pp. 1215–1219.

# Requirements
- Jupyter Notebook
- Keras
- numpy
- scipy
- matplotlib
- Pillow

# Data
fMRI dataset is provided from Ref. [7] (https://github.com/KamitaniLab/GenericObjectDecoding)*.

*Copyright : CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

[7] T. Horikawa and Y. Kamitani, “Generic decoding of seen and imagined objects using hierarchical visual features,” Nature Commun., vol. 8, no. 15037, pp. 1–15, 2017.

- candidate_name.txt : category names of candidates (10,000 candidates, top 50 categories are 50 test categories)
- test_images : images of the original 50 test categories
- subjectxx.mat : fMRI activity of subjectxx
  - subxx_train_sort : fMRI activity of subjectxx for training data
  - subxx_test_ave : fMRI activity of subjectxx for test data
- visual&semantic.mat : visual and semantic features 
  -  VGG19_train_sort : visual features for training data
  -  VGG19_candidate : visual features of candidate categories
  -  word2vec_train_sort : semantic features for training data
  -  word2vec_candidate : semantic features of candidate categories
  -  candidate_names : ImageNet ID of candidate categories

# Cite
Please cite the following papers if you want to use this code in your work.
```
@article{akamatsu2021perceived,
  title={Perceived Image Decoding From Brain Activity Using Shared Information of Multi-Subject fMRI Data},
  author={Akamatsu, Yusuke and Harakawa, Ryosuke and Ogawa, Takahiro and Haseyama, Miki},
  journal={IEEE Access},
  volume={9},
  pages={26593--26606},
  year={2021}
}
```
```
@inproceedings{akamatsu2020multi,
  title={Multi-view Bayesian Generative Model for Multi-Subject fMRI Data on Brain Decoding of Viewed Image Categories},
  author={Akamatsu, Yusuke and Harakawa, Ryosuke and Ogawa, Takahiro and Haseyama, Miki},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1215--1219},
  year={2020}
}
```
