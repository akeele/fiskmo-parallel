# fiskmo-parallel
This repository contains scripts to find parallel sentences from large amount of text. 

## Requirements
1. First you need to use Facebook research's [LASER](https://github.com/facebookresearch/LASER/tree/master/tasks/embed) library to embed your sentences. 
2. Next you will need to download [FAISS](https://github.com/facebookresearch/faiss)
3. [NumPy](numpy.org)

## Steps
1. Embed your text files using the [LASER](https://github.com/facebookresearch/LASER/tree/master/tasks/embed) library. Now you will have 1024 dimensional sentence vectors. This can take for a few days if you have tens of millions of sentences.
2. Create FAISS index. The index I'm using is "OPQ32_128,IVF32768,PQ32". You can read more about those in the [FAISS wiki](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index). The embeddings you have gotten from LASER.
```
$ python create_faiss_index.py --embedding-file EMBEDDINGS_FROM_LASER --index-name NAME --batch-size 8192 --training-size 1500000
```
3. Search embeddings from index. This step will perform kNN search for the embeddings you feed it from the previously created index. This script will output a score file containing euclidean distances and indices for the 10 nearest neighbors and margin score with the nearest neighbor. Note that this margin is different from the one used in LASER. We divide the distance of the nearest neighbor with the mean of the other 9. This way the one with the lower margin score should be better.
```
$ python search_laser_vectors.py --index YOUR_INDEX --embeddings EMBEDDINGS_YOU_WANT_TO_SEARCH --batch-size 16834 --output OUTPUT_FILE
```
4. You will fetch the parallel sentences from the original text files using the just created scores file. Source sentences are those that you previously searched from the FAISS index and target sentences are the indexed ones.
```
$ python fetch_parallel_sentences.py --score-file SCORE_FILE --source-sentences SOURCE_LANGUAGE_FILE --target-sentences TARGET_LANGUAGE_FILE --output OUTPUT_FILE
```
5. Finally sort the parallel sentences file. The file looks likes this:
```
0.092883  I have a dog. Minulla on koira
```
So we can sort it easily because the margin score is the first item on every line
```
$ sort YOUR_PARALLEL_SENTENCES_FILE > SORTED_FILE
```
