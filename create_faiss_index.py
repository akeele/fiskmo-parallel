import sys
import os
import numpy
import faiss
import argparse


DIMENSIONS = 1024


def read_embeddings(embeddings_file):
    with open(embeddings_file, 'rb') as embedding_file:
        file_size = os.fstat(embedding_file.fileno()).st_size
        while embedding_file.tell() < file_size:
            embedding = numpy.fromfile(embedding_file, numpy.float32, DIMENSIONS)
            yield embedding

def get_embeddings_batch(embeddings_file, batch_size):
    batch_embeddings = []
    batch_count = 0
    for embedding in read_embeddings(embeddings_file):
        batch_embeddings.append(embedding)
        if len(batch_embeddings) == batch_size:
            batch_embeddings = numpy.stack(batch_embeddings, axis=0)
            yield batch_embeddings
            batch_count += 1
            print("Batch number %i finished" % batch_count, end="\r") 
            batch_embeddings = []


def add_embeddings_to_index(embeddings, gpu_index, batch_size):
    for embeddings_batch in get_embeddings_batch(embeddings, batch_size):
        faiss.normalize_L2(embeddings_batch)
        gpu_index.add(embeddings_batch)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--embedding-file", help="File containing sentence embeddings")
    argparser.add_argument("--index-name", help="Index name to store")
    arguments = argparser.parse_args()
    
    BATCH_SIZE = 8192
    
    # We need between 983040 and 8388608 training vectors
    TRAINING_SIZE = 1500000
    training_embeddings = numpy.fromfile(arguments.embedding_file, numpy.float32, TRAINING_SIZE*DIMENSIONS)
    training_embeddings.resize(training_embeddings.shape[0] // DIMENSIONS, DIMENSIONS)
    
    print("%i training embeddings..." % training_embeddings.shape[0])
    
    faiss.normalize_L2(training_embeddings)
    print("Training embeddings normalized...")
    
    index = faiss.index_factory(DIMENSIONS, "OPQ32_128,IVF32768,PQ32")
    gpu_index = faiss.index_cpu_to_all_gpus(index)
    print("Index created...")
    
    gpu_index.train(training_embeddings)
    print("Index trained", gpu_index.is_trained)
    add_embeddings_to_index(arguments.embedding_file, gpu_index, BATCH_SIZE)
    print("%i embeddings in index" % gpu_index.ntotal)

    # Store index to a file
    index_to_store = faiss.index_gpu_to_cpu(gpu_index)
    print(index_to_store.ntotal)
    faiss.write_index(index_to_store, arguments.index_name)
