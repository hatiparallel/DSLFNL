# Overall Explanation
- This is an PyTorch implementation of "Deep Self-Learning From Noisy Labels" in ICCV2019. But it is not the original one, and it seems not provide the same performance as what is described in the original paper.
- This implementation is also explained in Japanese in [Qiita](https://qiita.com/hatiparallel/items/e79fb004f7ad687304c8).
- This is implemented on [my template](https://github.com/hatiparallel/template).

# Source Code
- main.py is the main file for training and test.
- train.py trains the model for 1 epoch.
- test.py tests the model accuracy.
- models.py defines the model for deep learning.
- criteria.py defines the loss function.
- loader.py loads the datasets and defines dataset classes.
- logger.py saves logs during training and test.
- correct.py corrects the label for deep self-learning.
- checklabel.py checks whether correct.py works well using a trained model.
- misc.py defines other functions.

# Execution
After you download and put the datasets in the appropriate place, please execute like this:
`$ python3 main.py --data Clothing1M --epochs 15 -c ccenoisy`
`$ python3 main.py --data Food101n --epochs 30 -c ccenoisy`