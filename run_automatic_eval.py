import argparse

from typing import Any, Callable, List
import numpy as np
import pandas as pd
import sys
import json
import os

import sacrebleu
from comet import download_model, load_from_checkpoint

sys.path.append(os.environ['MUDA_HOME'])
from muda.langs import create_tagger
from muda.metrics import compute_metrics

comet_metric = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
context_qe_metric = load_from_checkpoint(download_model("Unbabel/wmt20-comet-qe-da"))
context_qe_metric.enable_context()

def read_file(fname):
    output = []
    with open(fname) as f:
        for line in f:
            output.append(line.strip())
    return output

def recursive_map(func: Callable[[Any], Any], obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: recursive_map(func, v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_map(func, v) for v in obj]
    else:
        return func(obj)
    
# Referred from https://github.com/amazon-science/doc-mt-metrics/blob/main/Prism/add_context.py
def add_context(orig_txt: List[str], context_same: List[str], context_other: List[str], 
                sender_ids: List[str], sep_token: str = "</s>", ws: int = 2, ) -> List[str]:
    
    if not (len(orig_txt) == len(context_same)== len(context_other)):
        raise Exception(f'Lengths should match: len(orig_txt)={len(orig_txt)}, len(context_same)={len(context_same)}, len(context_other)={len(context_other)}')
    i = 0
    augm_txt = []
    for i in range(len(orig_txt)):
      context_window = []
      for j in range(max(0, i - ws), i):
        if sender_ids[j] == sender_ids[i]:
          context_window.append(context_same[j])
        else:
          context_window.append(context_other[j])
      augm_txt.append(" {} ".format(sep_token).join(context_window + [orig_txt[i]]))
    return augm_txt

def get_scores(df, batch_size=128, gpus=1, ws=2):
    df["comet_scores"] = comet_metric.predict([{"mt": y, "ref":z, "src": x} for x, y, z in
                                zip(df["source"].to_list(), df["mt"].to_list(), df["reference"].to_list())], 
                                batch_size=batch_size, gpus=gpus)['scores']

    source_with_context = add_context(df["source"].to_list(), df["source"].to_list(), df["mt"].to_list(), 
                                df["sender"].to_list(), context_qe_metric.encoder.tokenizer.sep_token, ws)
    hyp_with_context = add_context(df["mt"].to_list(), df["mt"].to_list(), df["source"].to_list(), 
                                df["sender"].to_list(), context_qe_metric.encoder.tokenizer.sep_token, ws)
    df["context_qe_scores"] = context_qe_metric.predict([{"mt": y, "src": x} for x, y in
                                zip(source_with_context, hyp_with_context)], batch_size=batch_size, gpus=gpus)['scores']
    
    for lp, lp_df in df.groupby(["source_language", "target_language"]):
        print(f"==== Scores for {lp} ===")
        print(f"chrF: {sacrebleu.corpus_chrf(lp_df['mt'].to_list(), [lp_df['reference'].to_list()]).score}")
        print(f"BLEU: {sacrebleu.corpus_bleu(lp_df['mt'].to_list(), [lp_df['reference'].to_list()]).score}")
        print(f"COMET-22: {np.mean(lp_df['comet_scores'])}")
        if lp[0] == "en":
            print(f"Contextual-Comet-QE: {np.mean(lp_df['context_qe_scores'])}")
        print("\n")


def get_muda_accuracy_score(srcs, refs, docids, tgt_lang="de", 
                  awesome_align_model="bert-base-multilingual-cased", 
                  awesome_align_cachedir=None, load_refs_tags_file=None,
                  cohesion_threshold=3, dump_hyps_tags_file=None, dump_refs_tags_file=None,
                  phenomena=["lexical_cohesion", "formality", "verb_form", "pronouns"],
                  hyps=None) -> None:

    tagger = create_tagger(tgt_lang,
        align_model=awesome_align_model,
        align_cachedir=awesome_align_cachedir,
        cohesion_threshold=cohesion_threshold,
    )
    
    if not load_refs_tags_file:
        preproc = tagger.preprocess(srcs, refs, docids)
        tagged_refs = []
        for doc in zip(*preproc):
            tagged_doc = tagger.tag(*doc, phenomena=phenomena)
            tagged_refs.append(tagged_doc)
    else:
        tagged_refs = json.load(open(load_refs_tags_file))

    preproc = tagger.preprocess(srcs, hyps, docids)
    tagged_hyps = []
    for doc in zip(*preproc):
        tagged_doc = tagger.tag(*doc, phenomena=phenomena)
        tagged_hyps.append(tagged_doc)

    tag_prec, tag_rec, tag_f1 = compute_metrics(tagged_refs, tagged_hyps)
    for tag in tag_f1:
        print(
            f"{tag} -- Prec: {tag_prec[tag]:.2f} Rec: {tag_rec[tag]:.2f} F1: {tag_f1[tag]:.2f}"
        )
    print()

    if dump_hyps_tags_file:
        with open(dump_hyps_tags_file, "w", encoding="utf-8") as f:
            json.dump(recursive_map(lambda t: t._asdict(), tagged_refs), f, indent=2)

    if not load_refs_tags_file and dump_refs_tags_file:
        with open(dump_refs_tags_file, "w", encoding="utf-8") as f:
            json.dump(recursive_map(lambda t: t._asdict(), tagged_refs), f, indent=2)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--hypothesis_file", type=str, required=True)

    # comet22 arguments
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--gpus", default=1, type=int)

    # context-comet-qe arguments
    parser.add_argument("--ws", default=2, type=int)


    # MuDA arguments as defined in https://github.com/CoderPat/MuDA/blob/main/muda/main.py
    parser.add_argument("--tgt-lang", type=str, required=True)
    parser.add_argument(
        "--dump_hyps_tags_file", 
        type=str, 
        default=None,
        help="If set, dumps the hypothesis tags to the specified file.")
    parser.add_argument(
        "--dump_refs_tags_file", 
        type=str, 
        default=None,
        help="If set, dumps the reference tags to the specified file.")
    parser.add_argument(
        "--load_refs_tags_file", 
        type=str, 
        default=None,
        help="If set, loads the reference tags from the specified file.")
    parser.add_argument(
        "--cohesion-threshold",
        default=3,
        type=int,
        help="Threshold for number of (previous) occurances to be considered lexical cohesion."
        "Default: 3",
    )
    parser.add_argument(
        "--phenomena",
        nargs="+",
        default=["lexical_cohesion", "formality", "verb_form", "pronouns"],
        help="Phenomena to tag. By default, all phenomena are tagged.",
    )
    parser.add_argument(
        "--awesome-align-model",
        default="bert-base-multilingual-cased",
        help="Awesome-align model to use. Default: bert-base-multilingual-cased",
    )
    parser.add_argument(
        "--awesome-align-cachedir",
        default=None,
        help="Cache directory to save awesome-align models",
    )
    args = parser.parse_args()
    return args


def main(args):
    df = pd.read_csv(args.input_csv)
    
    if args.hypothesis_file is not None:
        df["mt"] = read_file(args.hypothesis_file)

    # chrF, BLEU, Context-COMET-QE and COMET-22 scores
    get_scores(df, batch_size=args.batch_size, gpus=args.gpus, ws=args.ws)
    
    # MuDA accuracy score
    df = df[df.source_language=="en"]
    get_muda_accuracy_score(df["source"].to_list(),
                            df["reference"].to_list(),
                            df["doc_id"].to_list(),
                            hyps=df["mt"].to_list(),
                            tgt_lang=args.tgt_lang, 
                            awesome_align_model=args.awesome_align_model, 
                            awesome_align_cachedir=args.awesome_align_cachedir,
                            dump_hyps_tags_file=args.dump_hyps_tags_file,
                            dump_refs_tags_file=args.dump_refs_tags_file,
                            load_refs_tags_file=args.load_refs_tags_file,
                            phenomena=args.phenomena, 
                            cohesion_threshold=args.cohesion_threshold,)


if __name__ == "__main__":
    args = get_args()
    main(args)