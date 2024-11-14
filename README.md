# DeepMineLys

In this study, we developed a convolutional neural network (CNN)-based framework for phage lysin mining, hereafter referred to as DeepMineLys. We used two-track in DeepMineLys. It extracted convolution features from the tasks assessing protein embeddings (TAPE) and physicochemical (PHY) embeddings, respectively, resulting in a progressively better statistical summary, or representation, of the protein sequence. For convenience we named this vector representation simply ‘UniRep’ everywhere in the main text.

DeepMineLys outperforms the existing methods by almost a scale of magnitude, and suggests that applying deep learning to the mining of vast metagenomics data could provide a significantly expanded opportunity for medical, industrial, agricultural, and food applications of enzymes.

DeepMineLys was implemented using the Python package Keras (version 2.6.0) (https://keras.io/) with TensorFlow backend (version 2.6.0). Our model training was conducted on a workstation with dual Ubuntu 16.04.7 and NVIDIA GeForce RTX 2080Ti graphics-processing unit.

------

Software Name: DeepMineLys V1.0

Year: 2023

*Rights Statement*

```
All rights of the software and associated documentation are owned by Lin lab. This software is currently under application for software copyright registration in China with the application number 2023R11S2149910. All rights and intellectual property of this software are protected under law.
```

*Usage and Distribution*

```
No one is permitted to use, copy, modify, merge, publish, distribute, sublicense, or deal in the software in any way, commercially or non-commercially, without explicit written permission.
```

*Contacts*

201810107448@mail.scut.edu.cn; 202120124523@mail.scut.edu.cn; biyangxf@scut.edu.cn; zl.lin@siat.ac.cn



### 1. INSTALLATION

Standard (harder):

Requirements in Ubuntu:

* python >= 3.6.8

* gcc >= 7.3.0

* Tensorflow = 2.6.0

* Keras = 2.6.0

  ```
  # For sequence embedding:
  conda env create -f deepminelys.yml
  conda env create -f phy_environment.yml
  ```

  


* TAPE embedding: available at https://github.com/songlab-cal/tape

  ```
  # babbler-1900 (UniRep model)
  tape-embed unirep input.fasta output_filename.npz babbler-1900 --tokenizer unirep
  
  # And then you can run the read_npz.py to get the tape embedding
  ```

* PHY embedding: available at R pakage "Peptides" by 

  ```
  install_github("dosorio/Peptides")
  ```

  

### 2. DeepMineLys USAGE

```
# running on command line
python predict_model.py --input_path INPUT_PATH --model_path MODEL_PATH --output_path OUTPUT_PATH

For example, you can run DeepMinelys from within the DeepMineLys folder using the following command:
python predict_model.py --input_path ./Example/input_seq_tape_phy.csv --model_path ./prediction_model --output_path ./outputs/prediction.csv
```



### 3. Examples

```
# First input protein sequences file
# input.fasta
>seq_1
MAKVQFKPRATTEAIFVHCSATKPSQNVGVREIRQWHKEQGWLDVGYHFIIKRDGTVEAGRDELAVGSHAKGYNHNSIGVCLVGGIDDKGKFDANFTPAQMQSLRSLLVTLLAKYEGSVLRAHHDVAPKACPSFDLKRWWEKNELVTSDRG	
>seq_2
MPENVPRSGLYYWPINPRKARPDVRYLDPDYYLGVKRPDGSWLVPPGYWHTGVDLNGPGGGDTDCGQPVHAMTDGRVVIAGRFPVWGGMVVLWHPSAGVWTRYGHLRDILVHPGDVVVAGQQIGTIGKMTTGGYCHLHFDVFIRVPPASEGGWLFFPRGGEEARQKVLTYLVDPEAFLAKQAQAGRLREPPAWRTA
>seq_3
MTEPTDTPSTPEPVAPAAPAPAAPAPKSEDLPDWAREKLSKANTEAANYRVQLRTVETERDSLAEKLAALEAQAAQAATSASERQNDFDRLVTAVQALTPDPTPLFTFANTLQGDSEEALKTHAESLKTLFGLKNGPVAAVDRSQGLGTEAPSNDPAVAFTALMKNQLGK
>seq_4
MKERRQRLRKSGAFGGQRVRYFKILCIVLFASLLVACHQISSGTVVDKYIDEPHTTFIPVINGKSSVLVPTRTKRKYILVVSGFTGNKQVEERFEVTANEYKHYEIGNTFIQDAVLENKEEDKE
>seq_5
MTVMRIEQLVEMYIDNIYNYPYPYDEAMFNKKNEEIRPYVDNSLYKAIQRLRPYYDVIEYMNISKKDYNVKEVTTGQYEVDFEVTSVSKTNFNHYAESTFKETIKIQLNPIKIESLDDSATQHYATYQDLEDDYKYDLTLPHKASELVQKNINNKSIQYQFKGAPKDNPFESDTTSLLDSYNMVYWLYNDEETNLNYPLDYNSILNSGVFKDISVRHKYISDIDLLEDGDLLFFGKNNNEVGIYVGDKEYVTIKGKFPKDNTTIQKYNLEKDWELFNGKIYRLKDDYL
>seq_6
MAQPQIAANKTKALRIGQRLMVGAKSFGKGVRNSVDTSQQVITQSARKIKSDQKRINAENKKQERFEDAVREETERRQKSLTKGAYGVGSAAKKLTEKVMVDPMKAIWNIIAAWAIKNLPIIVDEVRKFVKKVRIVIAAINNAFRATGNLFKGWLSYGQAWLTNMVTFDWGDSTGRLEEAQAEIDNSYDELDASWNTIYNVWGKEEEELDKMLTWLDSGKTIKQATDAITQGIPLPQTPAFGDGSREGGGGYGGSGIQMSADEQQLTESLIAGEEGVRTKAYQDTEGIWTIGYGQTRINGRAVRPGDTITKAQALGGFRGALAEHQQRAINQLGEERWSQLDARSRAVLTSITYNYGSIPGRVLPAAKTGNAEDIAVAMNSLYGDNRGVLKGRRQREQSILRGGTSSYLDKDFMAGGQFAGSGTGPLVMGGGNESTSGSISSSGGTTNTTAMKRGDMVGGFSVSSAFGRRAAPTAGASSNHGGLDIATPQGTYLAFDVDVEIMFAGSAGGYGYVIDAWSASLGIQFRCAHMSVLMCNPGQKVRAGTAVGRTGGAVGSRGAGTSTGPHLHFEVSNQKGSANYGGSNSASMLARYAKHLILSSSKPQPQSLRPATVSSSSQQTANQLNSSASSRRTNGRRTDQKIILIKENTIIK
>seq_7
MQKPDGLYEVLNIVRVFYEHGIDEHLSVCLLIEMIGSDIVLGVSRAWAFHELSSFKFRKGLVSHLATALFVIIFYPFAIFMHLGSVIDTFIYAMMAAYGSSILANLSSLGVKFPYIDRYIRLNIDKEKFILLDEEEEEEND
>seq_8
MKKRKKKMINFKLRLQNKATLVALISAVFLMLQQFGLHVPNNIQEGINTLVGILVILGIITDPTTKGIADSERALSYIQPLDDKEVY
```

Format of Input file:

The input_seq_file is represented in a sequence file with specific formatting:

- File size: N × 1905, where N represents the number of sequences.
- The file consists of multiple rows, each containing the following components:
  - Column 1: Sequence name.
  - Columns 2 to 1901: TAPE embedding.
  - Columns 1902 to 1905: PHY embedding.

```
# input_seq_tape_phy.csv
seq_1,0.4757,0.482,0.5053,0.4481,0.3669,0.5189,0.3791,...,0.4777.4,0.4491.1,0.4732.2,0.7467160165049,4.450799961725889,-0.43443708609271503,77.4834437086093
seq_2,0.4973,0.5069,0.5056,0.4748,0.3633,0.5021,0.3436,...,0.3981,0.4945,0.4782,0.681948663202157,2.49431590626754,-0.29642857142857104,76.0204081632653
seq_3,0.4868,0.4542,0.4904,0.4504,0.4745,0.4678,0.5096,...,0.4519,0.5594,0.5473,0.610328043602646,-7.93617720907427,-0.49764705882352894,70.8823529411765
seq_4,0.5158,0.5037,0.5913,0.4964,0.5676,0.5072,0.3839,...,0.344,0.4701,0.5143,0.465177119730606,5.13562442998637,-0.364516129032258,87.90322580645159
seq_5,0.4766,0.4602,0.5086,0.466,0.4483,0.4831,0.3514,...,0.493,0.5284,0.4874,0.6530285153584701,-15.661930598998401,-0.7701388888888892,78.125
seq_6,0.4922,0.5336,0.5074,0.4363,0.3494,0.4843,0.5136,...,0.4999,0.5204,0.5631,0.8150300725595551,20.449946237806,-0.447166921898928,72.2511485451761
seq_7,0.532,0.5195,0.5539999999999999,0.5093,0.5438,0.575,0.4492,...,0.5696,0.5589,0.5103,0.626468764274431,-8.62963134384603,0.39361721276596,115.460992907801
seq_8,0.5606,0.544,0.5815,0.5423,0.5957,0.5938,0.4881,...,0.5108,0.5201,0.5335,0.566070600668809,5.036402725200539,0.141379310344828,124.367816091954

```

Format of output file:

- The predict output file size: N × 6, where N represents the number of sequences.

- The predict output file consists of multiple rows, each containing the following components:

  - Column 1: Sequence name.

  - Column 2 : Predict lable (0 for Non-EVH, 1 for Endolysin, 2 for VAL and 3 for Holin).

  - Column 3 to 6: Prediction score of Non-EVH, Endolysin, VAL and Holin, respectively.

    

```
# output_prediction_file
seq_1 1 0.0014717972371727228 0.9979894161224365 0.0005385022377595305 3.2607002253826067e-07
seq_2 1 1.8677676507650176e-06 0.999983549118042 1.4528728570439853e-05 1.8509032848057494e-11
seq_3 0 0.9999979734420776 5.141233714311966e-07 1.5903888197499327e-06 2.3045844399494086e-10
seq_4 1 0.35380280017852783 0.4793384075164795 0.13507451117038727 0.0317843072116375
seq_5 0 0.9947683811187744 0.0013339779106900096 0.003882663557305932 1.5075192095537204e-05
seq_6 2 9.467504423810169e-05 0.000488573219627142 0.9994168281555176 5.1991260185957344e-09
seq_7 3 0.24308863282203674 0.016076141968369484 0.02078438550233841 0.7200508117675781
seq_8 3 0.003980554174631834 4.724847531178966e-05 9.934492118190974e-05 0.9958729147911072
```
