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


## Evaluation 

We additionally release a scoring script to compute the automatic metrics as descibed in the [shared task page](https://www2.statmt.org/wmt24/chat-task.html).

To use the scoring script you need to install the following libraries:

1. Install MuDA ([Fernandes et al., 2021](https://aclanthology.org/2023.acl-long.36/)) and package requirements by:
```
    git clone git@github.com:CoderPat/MuDA.git
    cd MuDA
    pip install -r requirements.txt
```
2. Set MuDA path to `export MUDA_HOME=<path_to_muda>`
3. Install SacreBLEU ([Post 2018](https://aclanthology.org/W18-6319/)) using ```pip install sacrebleu```
4. Install COMET ([Rei et al., 2020](https://aclanthology.org/2020.emnlp-main.213/)) using:
```pip install git+https://github.com/Unbabel/COMET.git```


Usage:

```python run_automatic_eval.py --input_csv shared_task_valid/en-de.csv --hypothesis_file shared_task_valid/en-de.baseline.txt --tgt-lang de```

## License

Please note, that all the data released for the WMT24 Chat Translation task is under the license of [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) and can be freely used for research purposes only. Please note that, as the license states, no commercial uses are permitted for this corpus. We just ask that you cite the WMT24 Chat Translation Task overview paper. Any other use is not permitted unless previous written authorization is given by Unbabel.
