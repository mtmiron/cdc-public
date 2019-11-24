# CDC text classification challenge

Result: 8th place out of ~95.

tmp.txt is the list of ensemble combinations that were tested; caching the logits got the speed up to 120 ensembles per second.  That only worked because "ensemble," as used here, was nothing more than the summation of the model's logits averaged.  Score went from 0.885 to 0.890.  GPU RAM and time prevented me from doing much else, and I had the weights for dozens of models saved.

See augmentation_techniques.txt for augmentation stuff that ended up not helping.

Models
------
- xlnet: didn't score particularly well, no matter what I tried.  It's possible I needed more than 5-7 epochs, but I never went past that many.  Theoretically it should have beaten roberta.  ~0.88 after 24 hours of training.
- bert: scored about the same as xlnet.  ~0.88 after ~12 hours of training.
- roberta: consistently scored the best.  3-5 epochs was the sweet spot. ~0.885 after ~12 hours of training.
- distilbert: scored almost as well as roberta in a fraction of the time. ~0.878 after ~2-4 hours of training.
- fastText: what can I say... 90% of the transformer's accuracy in the blink of an eye.  Scored ~0.84 consistently after 2-10 minutes (!!) of training, but that was only my local scoring (I didn't bother submitting this one's solution).  I walked away from this challenge deciding that I'll pretty much always use fastText unless I'm trying to win something.

Files
-----
- main.py: train and/or predict using individual models.
- ensemble.py: use models from main.py in a trivial ensemble-fashion.
- simpletransformersmodel.py: modification of simpletransformers.
- distilbertmod.py: modification of Huggingface's distilbert model (junk that didn't help).
- robertamod.py: same as distilbertmod.py.
- ensemble.txt: the list of models I ended up using for my submission(s).
- (xxx).txt: output of stuff I tried.
- (xxx).csv: various train/validation splits of the provided training samples.

There's nothing of real note here.
