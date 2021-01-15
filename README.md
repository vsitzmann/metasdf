# MetaSDF: Meta-learning Signed Distance Functions
### [Project Page](https://vsitzmann.github.io/metasdf) | [Paper](https://arxiv.org/abs/2006.09662) | [Data]()
[Vincent Sitzmann](https://vsitzmann.github.io/)\*,
[Eric Ryan Chan](http://alexanderbergman7.github.io)\*,
[Richard Tucker](),
[Noah Snavely](http://www.cs.cornell.edu/~snavely/)<br>
[Gordon Wetzstein](https://stanford.edu/~gordonwz/)<br>
\*denotes equal contribution

This is the official implementation of the paper "MetaSDF: Meta-Learning Signed Distance Functions".

In this paper, we show how we may effectively learn a prior over implicit neural representations using
gradient-based meta-learning. 

While in the paper, we show this for the special case of SDFs with the ReLU nonlinearity, this works formidably well 
with other types of neural implicit representations - such as our work "SIREN"!

We show you how in our Colab notebook:

[![Explore MetaSDF in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vsitzmann/metasdf/blob/master/MetaSDF.ipynb)<br>

## DeepSDF
A large part of this codebase (directory "3D") is based on the code from the terrific paper "DeepSDF" - check them out!

## Get started
If you only want to experiment with MetaSDF, we have written a colab that doesn't require installing anything,
and goes through a few other interesting properties of MetaSDF as well - for instance, it turns out you can train
SIREN to fit any image in only just three gradient descent steps!

If you want to reproduce all the experiments from the paper, you can then set up a conda environment with all dependencies like so:
```
conda env create -f environment.yml
conda activate metasdf
```

## 3D Experiments

**Dataset Preprocessing**

Before training a model, you'll first need to preprocess the training meshes. Please follow the preprocessing steps used by DeepSDF if using ShapeNet.

**Define an Experiment**

Next, you'll need to define the model and hyperparameters for your experiment. Examples are given in 3D/curriculums.py, but feel free to make modifications. Although not present in the original paper, we've included some curriculums with positional encodings and smaller models. These generally perform on par with the original models but require much less memory.

**Train a Model**

After you've preprocessed your data and have defined your curriculum, you're ready to start training! Navigate to the 3D/scripts directory and run

`python run_train.py <curriculum name>`.

If training is interupted, pass the flag `--load` flag to continue training from where you left off.

You should begin seeing printouts of loss, with a summary at every epoch. Checkpoints and Tensorboard summaries are saved to the `'output_dir'` directory, as defined in your curriculum. We log raw loss, which is either the composite loss or L1 loss, depending on your experiment definition, as well as a 'Misclassified Percentage'. The 'Misclassified Percentage' is the percentage of samples that the model incorrectly classified as inside or outside the mesh.

**Reconstructing Meshes**

After training a model, recontruct some meshes using

`python run_reconstruct.py <curriculum name> --checkpoint <checkpoint file name>`.

The script will use the `'test_split'` as defined in the curriculum.

**Evaluating Reconstructions**

After reconstructing meshes, calculate Chamfer Distances between reconstructions and ground-truth meshes by running

`python run_eval.py <reconstruction dir>`.


## Torchmeta
We're using the excellent [torchmeta](https://github.com/tristandeleu/pytorch-meta) to implement hypernetworks.

## Citation
If you find our work useful in your research, please cite:
```
       @inproceedings{sitzmann2019metasdf,
            author = {Sitzmann, Vincent
                      and Chan, Eric R.
                      and Tucker, Richard
                      and Snavely, Noah
                      and Wetzstein, Gordon},
            title = {MetaSDF: Meta-Learning Signed
                     Distance Functions},
            booktitle = {Proc. NeurIPS},
            year={2020}
       }
```

## Contact
If you have any questions, please feel free to email the authors.
