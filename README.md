# slots-and-intents prediction


## Goal: Intents and Slots Prediction Model
This is a task-orientated semantic parsing problem based on a subset of TOPv2 (Task-Oriented Parsing v2) Dataset.
Your task is to create a semantic parsing model that correctly predicts intents and the relevant slot information from a natural language sentence. 

There is a notebook `exercise_boilerplate.ipynb` that provides some boilerplate code to get you going :) 


## What we are looking for:
- Can you create a basic baseline model? Why have you chosen this implementation? 
- What are the steps have you taken (please make sure your thinking is well commented) 
- Try to experiment with methods to improve your basic baseline architecture 
- How are you evaluating & comparing your training runs? 
- Given more time what would you do? 
    - architecture strategies
    - relevant research papers

### Submission
- Check-in your solution to a new branch and create a PR (`!git checkout -b 'submission'`)
- Please make sure to include your predictions using two files called **data/submission_mce.tsv** and **data/submission_ccf.tsv**. The former should contain the prediction of the model trained with cross-entropy loss and the latter file containing the results of training with a custom cost function written by you. Complete instructions can be found in the provided notebook. 


### Submission File Format
You should submit two `.tsv` files with exactly 1327 entries each. Your submission will show an error if you have extra columns or rows.
Each file should have exactly 1 column which contains a list of predicted intents and the relevant slot information:

```
[IN:GET_INFO_TRAFFIC traffic update for [SL:LOCATION 880 north ] please ]
[IN:GET_INFO_TRAFFIC what is traffic like in [SL:LOCATION china ] ]
[IN:UNSUPPORTED_NAVIGATION is there a carpool lane ahead ]
...
```

You can see an example submission file (example_submission.tsv) in the `data` folder.


## The TOPv2 (Task Oriented Parsing v2) Dataset


Slots and Intents are a hierarchical representation, similar to a syntax tree, where the intent is at the root of each tree. These trees can be nested to allow for more complex queries.
The following rules apply: 
1) The top-level node must be an intent, 
1) An intent can have tokens and/or slots as children, 
1) A slot can have either tokens as children or one intent as a child.

####  Simple Example (one intent): 
*Input text:* "Play	the	song	don't	stop	believin	by	Journey"
```
IN:PlaySongIntent
 ├── SL:SongName(don't stop believin)
 └── SL:ArtistName(Journey)
```
This example shows a simple single intent at the root of the tree, with two slots as children.

####  More Complex Example (multi intents): 
slots and intents can be nested as per this example: 

*Input text:* "What 's happening in Harvard Square this weekend"

```
IN:GET_EVENT
 ├── SL:LOCATION
 │   └──IN:GET_LOCATION
 │       └── SL:POINT_ON_MAP(Harvard Square)
 └── SL:DATE_TIME(this weekend)
```
This shows a more complex example of a nested intent within a slot. 

###  Dataset Stats
The provided dataset is a subset of TOPv2 dataset that was first released by Sonal Gupta et al. in the 2018 paper *Semantic Parsing for Task-Oriented Dialog using Hierarchical Representations*.
- 6817 annotations, randomly split into 4796 training, 694 validation, and 1327 test utterances
- 13 intents and 27 slots
- The median (mean) depth of the trees is ~2, and the median (mean) length of the utterances is ~8 tokens.
29% of trees have a depth of more than 2. 

The columns are: 
- raw_utterance 
- TOP-representation: where the TOP-representation is an annotated version of the utterance


### Metrics
The model is evaluated with the following metrics, as defined in the paper and code in `semantic_parsing_dialog/evaluate.py`:
- Exact match fraction: the number of utterances whose full trees are correctly predicted
- Tree validity fraction: the percentage of predictions which formed valid trees (via bracket matching).
- Labelled bracketing scores (precision, recall, f1)

## Set up with GPU

**Local**

You can complete the assignment on your local machine if you have a GPU. We recommend at least 8 GB of VRAM to train the model within a reasonable amount of time.
