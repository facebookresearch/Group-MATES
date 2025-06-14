# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024 mlfoundations
#
# This code is based on https://github.com/mlfoundations/dclm

from .enrichers import line_counter_enricher
from .language_id_enrichers import detect_lang_paragraph_enricher, detect_lang_whole_page_enricher
from .quality_prediction_enrichers_calc_fasttext import classify_fasttext_hq_prob_enricher
from .quality_prediction_enrichers_kenlm_model import ken_lm_perplexity_enricher
from .dim_enrichers import assign_dim_score_enricher