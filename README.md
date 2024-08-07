# WMT 2024 Chat Shared Task

This repository contains the dataset for the chat shared task organized with WMT 2024.

## Data

The dataset is provided in a csv format, with each row specifying the source language, target language, source, reference, document id, the sender information and the client id.

Table 1: Number of source segments in the released dataset.

|language pair	|train	| valid | test |
|---	|---	|--- |--- |
|EN <-> DE	| 17805| 2569 | 2041 |
|EN <-> FR	| 15027	| 3007 | 2091 |
|EN <->PT-BR	| 15092	| 2550 | 2040 |
|EN <-> KO	  | 16122	| 1935 | 1982 |
|EN <-> NL	  | 15463	| 2549 | 2015 |

## Submission Instructions
- The translations submission deadline is the **26th of July**, End-of-day, anywhere on earth.
- Submissions should be emailed to the address wmt.chat.task@gmail.com with the subject line: `WMT 2024 CHAT Shared Task Submission`. Submissions should be packed in a compressed file with the following naming convention: `submission_[team-name].tar.gz`. Packages should be organized by language pair (e.g. `./en_de/`).Each directory should include:
1. One plaintext output file with one sentence per line for each system submitted, pre-formatted for scoring (detokenized, detruecased), for that language pair, named as `[system-id].txt`. The order of sentences should be the same as the original csv files provided for each language pair. You can submit at most 3 system outputs per language pair, one `primary` and up to two `contrastive`. 
2. Note that each file will include translations for both `x->y` and `y->x` language directions. However, the final automatic evaluation scores will be reported per language direction.
3. A readme file with the name of the team and the participants, a brief description of each system(s) submitted including details about training data and architecture used, including which system they prefer to be used for final evaluation by the organizers, and the contact persons' name and email.
- We invite participants to submit a paper describing their system(s) via the conference submission page. The paper submission deadline for WMT is the **20th of August**.

## Baselines 

Our baseline system uses [NLLB-3.3B](https://huggingface.co/facebook/nllb-200-3.3B) model to generate translations. The quality of translations using automatic metrics are provided below:

From Engish -> XX

|language pair	|chrF	| COMET | 
|---	|---	|---	|
|EN -> DE	| 66.24	| 86.76	|
|EN -> FR	| 74.31	| 88.85 |
|EN ->PT-BR	| 60.68	| 87.38	|
|EN -> KO	  | 30.47	| 84.59 |
|EN -> NL	  |	60.37| 87.29	|

From XX -> English
|language pair	|chrF	| COMET | 
|---	|---	|---	|
|DE -> EN	| 65.92	| 85.88 | 
|FR	-> EN | 72.53	| 85.44 |
|PT-BR -> EN	| 67.13	| 86.83 | 
|KO -> EN	  |	54.47 |82.81 |
|NL -> EN	  |	 66.90| 86.58 |  

## Evaluation 

We additionally release a scoring script to compute the automatic metrics as descibed in the [shared task page](https://www2.statmt.org/wmt24/chat-task.html).

To use the scoring script you need to install the following libraries **in the following order**:

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
## Registration
If you are participating in the task, make sure to register your team along with the language pairs you intend to participate in using this [registration form](https://forms.gle/zVFtGpt92uvC6XSS9) 

## License

Please note, that all the data released for the WMT24 Chat Translation task is under the license of [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) and can be freely used for research purposes only. Please note that, as the license states, no commercial uses are permitted for this corpus. We just ask that you cite the WMT24 Chat Translation Task overview paper. Any other use is not permitted unless previous written authorization is given by Unbabel.
