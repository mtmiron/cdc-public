# CDC text classification challenge

Result: 8th place out of ~95.

Bear in mind that these files were not written to be extensible, elegant, or shining examples of good code.  Frankly the code I write during a timed marathon contest isn't the sort of thing I'd like to be judged by, but if it's useful to you somehow, here it is.  *Note that all scores are from memory*

tmp.txt is the list of ensemble combinations that were tested; caching the logits got the speed up to 120 ensembles per second.  That only worked because "ensemble," as used here, was nothing more than the summation of the model's logits averaged.  Score went from 0.885 to 0.890.  GPU RAM and time prevented me from doing much else, and I had the weights for dozens of models saved.

See augmentation_techniques.txt for augmentation stuff that ended up not helping.

Models
------
All models except ERNIE2.0 and fastText were the huggingface implementations.  For fastText I used Facebook's implementation, and ERNIE was a slightly modified PaddlePaddle version.

If you want the 5 gigabyte zip of the 5 models I ended up submitting: https://drive.google.com/open?id=1NQBAMgR_FtlN_KOPBNoFtaASwT77Nhlz

- xlnet: didn't score particularly well, no matter what I tried.  It's possible I needed more than 5-7 epochs, but I never went past that many.  Theoretically it should have beaten roberta.  ~0.88 after 24 hours of training.
- bert: scored about the same as xlnet.  ~0.88 after ~12 hours of training.
- roberta: consistently scored the best.  3-5 epochs was the sweet spot. ~0.885 after ~12 hours of training.
- distilbert: scored almost as well as roberta in a fraction of the time. ~0.878 after ~2-4 hours of training.
- ernie: scored around ~0.68; it's conceivable I screwed up the implementation.
- fastText: what can I say... 90% of the transformer's accuracy in the blink of an eye.  Scored ~0.84 consistently after 2-10 minutes of training, but that was only my local scoring (I didn't bother submitting this one's solution).

Files
-----
- aspell.py: sorta-kinda useful interface to GNU's Aspell spell checker.
- main.py: train and/or predict using individual models.
- ensemble.py: use models from main.py in a trivial ensemble-fashion.
- simpletransformersmodel.py: modification of simpletransformers.
- distilbertmod.py: modification of Huggingface's distilbert model (junk that didn't help).
- robertamod.py: same as distilbertmod.py.
- ensemble.txt: the list of models I ended up using for my submission(s).
- (xxx).txt: output of stuff I tried.
- (xxx).csv: various train/validation splits of the provided training samples.

Notes
-----
There's nothing of real note here, but I do find it rather interesting that when using a Roberta model, randomly permuting the order of words in sentences only dropped my score from ~0.85 to ~0.70.  I assumed that's because classification depended mostly on specific keywords as opposed to phrases or combinations of words/subwords, but since the transformers are bi-directional, many of the sentences may have ended up with similar token sequences even after permuting.  I wasn't being vigilant about things I tried: it was an interesting tidbit from my perspective.
