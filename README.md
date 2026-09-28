# Does Less Fragmentation Mean Better Language Modeling?

Code and experiment records accompanying the paper:

**Does Less Fragmentation Mean Better Language Modeling? Evidence from Low-Resource African Languages**

**Eric KAMBIRE**

## Overview

This repository contains code and experiment records for a controlled comparison
of SentencePiece BPE and Unigram tokenization on Yorùbá, Luganda, and Swahili.

The study asks whether lower token fragmentation necessarily leads to better
held-out language modeling under a fixed token-level training budget.

## Main result

At vocabulary size 1,000, Unigram obtains lower held-out bits per character
than BPE for all three languages across three model-training seeds.

In the Yorùbá vocabulary-size analysis, BPE has lower token fertility while
Unigram has lower held-out BPC at vocabulary sizes 500, 1,000, and 2,000.

## Motivation

This work is a first step toward understanding how representations should be
designed for low-resource languages when data and compute are scarce.

The broader motivation includes African languages with strong oral traditions
and limited digital written resources. The experiments reported here are text-based.

## Data

The experiments use MasakhaNER:
https://github.com/masakhane-io/masakhane-ner

The dataset itself is not redistributed in this repository.

## Reproducibility

See results/experiment_manifest.json for the complete experimental protocol.

## Author

Eric KAMBIRE
Independent Researcher
eric.kambire@centralesupelec-alumni.com
