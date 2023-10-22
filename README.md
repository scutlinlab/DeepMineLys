# DeepMineLys

In this study, we developed a convolutional neural network (CNN)-based framework for phage lysin mining, hereafter referred to as DeepMineLys. We used two-track in DeepMineLys. It extracted convolution features from the tasks assessing protein embeddings (TAPE) and physicochemical (PHY) embeddings, respectively, resulting in a progressively better statistical summary, or representation, of the protein sequence. For convenience we named this vector representation simply ‘UniRep’ everywhere in the main text.

DeepMineLys outperforms the existing methods by almost a scale of magnitude, and suggests that applying deep learning to the mining of vast metagenomics data could provide a significantly expanded opportunity for medical, industrial, agricultural, and food applications of enzymes.

DeepMineLys was implemented using the Python package Keras (version 2.6.0) (https://keras.io/) with TensorFlow backend (version 2.6.0). Our model training was conducted on a workstation with dual Ubuntu 16.04.7 and NVIDIA GeForce RTX 2080Ti graphics-processing unit.

------

### Notice:

### INSTALLATION in Ubuntu

Standard (harder):

Requirements:

* python = 3.6.8

* Tensorflow = 2.6.0

* gcc = 7.3.0

* Keras = 2.6.0


* TAPE embedding: available at https://github.com/songlab-cal/tape

  ```
  babbler-1900 (UniRep model)
  tape-embed unirep my_input.fasta output_filename.npz babbler-1900 --tokenizer unirep
  ``` 

* PHY embedding: available at R pakage "Peptides" by 

  ```
  install_github("dosorio/Peptides")
  ```

  
### DeepMineLys USAGE: running on command line

```
python predict_model.py input_seq_file.csv model_dir_path
The input_seq_file is represented in a sequence file with specific formatting:
```
- File size: N * 1905, where N represents the number of sequences.
- The file consists of multiple rows, each containing the following components:
  - Column 1: Sequence name.
  - Columns 2 to 1901: TAPE embedding.
  - Columns 1902 to 1905: PHY embedding.





