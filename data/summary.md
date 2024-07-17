## Tasks in Lifelong ICL

Quick Link: [Meta-Train](#meta-train), [Meta-Test-NLI](#meta-test-nli), [Meta-Test-Domain](#meta-test-domain), [Meta-Test-RAFT](#meta-test-raft)

### Meta-Train
| No. | Name | Class | HF | Reference | License | Comment |
| ------- | ------- | --------------- | ----------- | ----------- | ----------- | ----------- | 
0 | ag_news | 4 | [Link](https://huggingface.co/datasets/ag_news) | [Paper](https://papers.nips.cc/paper_files/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html) | [Unspecified](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) | Topic |
1 | boolq | 2 | [Link](https://huggingface.co/datasets/google/boolq) | [Paper](https://aclanthology.org/N19-1300/) | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | Yes/No |
2 | emotion | 6 | [Link](https://huggingface.co/datasets/dair-ai/emotion) | [Paper](https://aclanthology.org/D18-1404/) | Unspecified | Emotion
3 | qqp | 2 | [Link](https://huggingface.co/datasets/nyu-mll/glue) | [Link](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) | Unspecified | Duplicate | 
48 | pragmeval_verifiability | 2 | [Link](https://huggingface.co/datasets/sileod/pragmeval) | [Paper](https://aclanthology.org/2022.lrec-1.255/), [Paper](https://aclanthology.org/W14-2105/) | Unspecified | - |
49 | beaver_tails | 2 | [Link](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | [Paper](https://arxiv.org/abs/2307.04657), [Github](https://github.com/PKU-Alignment/beavertails) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) | - |
50 | covid_fake_news | 2 | [Link](https://huggingface.co/datasets/nanyy1025/covid_fake_news), [Web](https://constraint-shared-task-2021.github.io/) |  [Paper](https://arxiv.org/abs/2011.03327) | Unspecified | - |
51 | lexical_rc_cogalexv | 6 | [Link](https://huggingface.co/datasets/relbert/lexical_relation_classification) | [Paper](https://aclanthology.org/W16-5309/) | Unspecified | - |
52 | lexical_rc_root09 | 3 | [Link](https://huggingface.co/datasets/relbert/lexical_relation_classification) | [Paper](https://aclanthology.org/L16-1722/) | Unspecified | - |
53 | pun_detection | 2 | [Link](https://huggingface.co/datasets/frostymelonade/SemEval2017-task7-pun-detection) | [Paper](https://aclanthology.org/S17-2005/), [Web](https://alt.qcri.org/semeval2017/task7/index.php?id=data-and-resources) | [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/deed.en) | - | 
54 | wiki_hades | 2 | [Link](https://huggingface.co/datasets/tasksource/wiki-hades) | [Paper](https://aclanthology.org/2022.acl-long.464/), [Github](https://github.com/microsoft/HaDes) | [MIT](https://github.com/microsoft/HaDes?tab=MIT-1-ov-file#readme) | - |
55 | i2d2 | 2 | [Link](https://huggingface.co/datasets/tasksource/I2D2) | [Paper](https://aclanthology.org/2023.acl-long.535/) [Link](https://i2d2.allen.ai/) | Unspecified | Commonsense |
56 | this_is_not_a_dataset | 2 | [Link](https://huggingface.co/datasets/HiTZ/This-is-not-a-dataset) | [Paper](https://aclanthology.org/2023.emnlp-main.531/) | [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) | - |
57 | clickbait | 2 | [Link](https://huggingface.co/datasets/marksverdhei/clickbait_title_classification) | [Paper](https://arxiv.org/abs/1610.09786) | Unspecified | - |
58 | amazon_massive_scenario | 18 | [Link](https://huggingface.co/datasets/SetFit/amazon_massive_scenario_en-US) | [Paper](https://arxiv.org/abs/2204.08582), [Github](https://github.com/alexa/massive) | [CC-BY 4.0](https://choosealicense.com/licenses/cc-by-4.0/) | - |
59 | environmental_claims | 2 | [Link](https://huggingface.co/datasets/climatebert/environmental_claims) | [Paper](https://arxiv.org/abs/2110.12010) | [CC-BY-NC-SA 4.0](https://spdx.org/licenses/CC-BY-NC-SA-4.0) | - |
60 | climate_commitments_actions | 2 | [Link](https://huggingface.co/datasets/climatebert/climate_commitments_actions) | [Paper](https://arxiv.org/abs/2110.12010) | [CC-BY-NC-SA 4.0](https://spdx.org/licenses/CC-BY-NC-SA-4.0) | - |
61 | tcfd_recommendations | 5 | [Link](https://huggingface.co/datasets/climatebert/tcfd_recommendations) | [Paper](https://arxiv.org/abs/2110.12010) | [CC-BY-NC-SA 4.0](https://spdx.org/licenses/CC-BY-NC-SA-4.0) | - |
62 | is_humor | 2 | [Link](https://huggingface.co/datasets/Blablablab/SOCKET) | [Paper](https://aclanthology.org/2023.emnlp-main.699/), [Paper](https://aclanthology.org/2021.semeval-1.118/)| Unspecified | - |
63 | brag_action | 2 | [Link](https://huggingface.co/datasets/Blablablab/SOCKET) | [Paper](https://aclanthology.org/2023.emnlp-main.699/), [Paper](https://aclanthology.org/2022.acl-long.273/) | Unspecified | - |

### Meta-Test-NLI

| No. | Name | Class | HF | Reference | License | Comment |
| ------- | ------- | --------------- | ----------- | ----------- | ----------- | ----------- | 
0 | rte | 2 | [Link](https://huggingface.co/datasets/super_glue) | [Paper 1](https://link.springer.com/chapter/10.1007/11736790_9) [2](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=33f25fae10da978fad3f48eb6bded2f733b28e92) [3](https://aclanthology.org/W07-1401/) [4](https://tac.nist.gov/publications/2009/additional.papers/RTE5_overview.proceedings.pdf) | Unspecified | -
1 | cb | 3 | [Link](https://huggingface.co/datasets/super_glue) | [Paper](https://semanticsarchive.net/Archive/Tg3ZGI2M/Marneffe.pdf), [Github](https://github.com/mcdm/CommitmentBank) | Unspecified | - |
2 | vitamin | 3 | [Link](https://huggingface.co/datasets/tals/vitaminc) | [Paper](https://aclanthology.org/2021.naacl-main.52/), [Github](https://github.com/TalSchuster/VitaminC) | [CC-BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/) | Fact Checking |
3 | qnli | 2 | [Link](https://huggingface.co/datasets/nyu-mll/glue) | [Paper](https://aclanthology.org/D16-1264/) | [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode) | SQuAD |
4 | wnli | 2 | [Link](https://huggingface.co/datasets/nyu-mll/glue) | [Paper](https://cs.nyu.edu/~davise/papers/WSKR2012.pdf), [Link](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html) | [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Winograd |
5 | sick | 3 | [Link](https://huggingface.co/datasets/sick) | [Paper](https://aclanthology.org/L14-1314/) | [CC-BY-NC-SA 3.0](https://spdx.org/licenses/CC-BY-NC-SA-3.0) | - |
6 | mnli | 3 | [Link](https://huggingface.co/datasets/nyu-mll/glue) | [Paper](https://aclanthology.org/N18-1101/) | [OANC](https://www.anc.org/OANC/license.txt), [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/deed.en), [CC-BY 3.0](https://creativecommons.org/licenses/by/3.0/deed.en) | Multi-Genre
7 | tomi_nli | 2 | [Link](https://huggingface.co/datasets/tasksource/tomi-nli) | [Paper](https://aclanthology.org/D19-1598/), [Link](https://arxiv.org/abs/2301.05948), [Github](https://github.com/facebookresearch/ToMi) | [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | Theory of mind

### Meta-Test-Domain

| No. | Name | Class | HF | Reference | License | Comment |
| ------- | ------- | --------------- | ----------- | ----------- | ----------- | ----------- | 
0 | rct20k | 5 | [Link](https://huggingface.co/datasets/armanc/pubmed-rct20k) | - | - | Biomedical |
1 | medical_question_pairs | 2 | [Link](https://huggingface.co/datasets/medical_questions_pairs) | - | - | Biomedical |
2 | acl_arc | 5 | [Link](https://huggingface.co/datasets/hrithikpiyush/acl-arc) | - | - | Science |
3 | scierc | 7 | [Link](https://huggingface.co/datasets/hrithikpiyush/scierc) | - | - | Science |
4 | function_of_decision_section | 7 | [Link](https://huggingface.co/datasets/nguha/legalbench) | - | - | Legal |
5 | sara_entailment | 2 | [Link](https://huggingface.co/datasets/nguha/legalbench) | - | - | Legal |
6 | stance_abortion | 3 | [Link](https://huggingface.co/datasets/tweet_eval) | - | - | Social Media |
7 | stance_feminist | 3 | [Link](https://huggingface.co/datasets/tweet_eval) | - | - | Social Media