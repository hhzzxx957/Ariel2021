# The DeepBlueAI Solution for Ariel-challenge
Repository hosting the solution for [Ariel data challenge 2021](https://www.ariel-datachallenge.space/) from team DeepBlueAI.

This code makes use of [numpy](https://github.com/numpy/numpy) and [pytorch](https://github.com/pytorch/pytorch).

## Repository content

- ```train.py``` contains main functions of training
- ```prediction.py``` contains main functions of inference
- ```model.py``` contains pytorch model codes
- ```utils.py``` includes code for generating dataset

## Baseline solution

This solution mainly based on 1D CNN, with additional planet features.
Compared to baseline, following things have been optimized:

### Accuracy:
1. Dilated multi-layer 1D CNN
2. L1 loss
3. Deeper MLP network
4. Add planet features
5. Add max-to-min signal features
6. Moving average smooth
7. 10 fold cross validation on full dataset
   
### Speed:
1. 1-cycle scheduler with adamw optimizer
2. preprocess files and save as a single pt file
3. Multi-gpu training

Final score has been improved from 9617 -> 9911

Training time on full dataset has been reduced from 30 hours to 15 minutes

