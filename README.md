# WMT 2024 Chat Shared Task

This repository contains the dataset for the chat shared task organized with WMT 2024.

## Data

The dataset is provided in a csv format, with each row specifying the source language, target language, source, reference, document id, the sender information and the client id.

Table 1: Number of source segments in the released dataset.

|language pair	|train	|
|---	|---	|
|EN <-> DE	|2778426	|
|EN <-> FR	|2151210	|
|EN <->PT	|2157768	|
|EN <-> KO	  |2287969	|
|EN <-> NL	  |2057627	| 


## Baselines 

Our baseline system uses [NLLB-3.3B](https://huggingface.co/facebook/nllb-200-3.3B) model to generate translations. The quality of translations using automatic metrics are provided below:

From Engish -> XX

|language pair	|chrF	| COMET | 
|---	|---	|---	|
|EN -> DE	| 66.24	| 86.76	|
|EN -> FR	| 74.31	| 88.85 |
|EN ->PT	| 60.68	| 87.38	|
|EN -> KO	  | 30.47	| 84.59 |
|EN -> NL	  |	60.37| 87.29	|

From XX -> English
|language pair	|chrF	| COMET | 
|---	|---	|---	|
|DE -> EN	| 65.92	| 85.88 | 
|FR	-> EN | 72.53	| 85.44 |
|PT -> EN	| 67.13	| 86.83 | 
|KO -> EN	  |	54.47 |82.81 |
|NL -> EN	  |	 66.90| 86.58 |  

## Evaluation 

We additionally release a scoring script to compute the automatic metrics as descibed in the [shared task page](https://www2.statmt.org/wmt24/chat-task.html).

To use the scoring script you need to install the following libraries in the following order:

1. Install MuDA ([Fernandes et al., 2021](https://aclanthology.org/2023.acl-long.36/)) and package requirements by:
```
    git clone https://github.com/CoderPat/MuDA.git
    pip install allennlp==2.10.0 sacremoses==0.0.53 spacy==3.3.0 spacy_stanza==1.0.2
```
2. Set MuDA path to `export MUDA_HOME=<path_to_muda>`
3. Install SacreBLEU ([Post 2018](https://aclanthology.org/W18-6319/)) using ```pip install sacrebleu```
4. Install COMET ([Rei et al., 2020](https://aclanthology.org/2020.emnlp-main.213/)) using:
```pip install git+https://github.com/Unbabel/COMET.git```


Usage:

```
for lp in en-de en-fr en-pt en-ko en-nl; do
    python run_automatic_eval.py --input_csv valid/${lp}.csv --hypothesis_file valid/${lp}.baseline.txt --tgt-lang ${lp: -2}
done
```

## License

Please note, that all the data released for the WMT24 Chat Translation task is under the license of [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) and can be freely used for research purposes only. Please note that, as the license states, no commercial uses are permitted for this corpus. We just ask that you cite the WMT24 Chat Translation Task overview paper. Any other use is not permitted unless previous written authorization is given by Unbabel.
