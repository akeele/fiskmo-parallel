import os
import faiss
import numpy
import argparse

DIMENSIONS = 1024
K = 10

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

def search_embeddings_from_index(index, embeddings):
    similarities, neighbor_ids = index.search(embeddings, K)
    return similarities, neighbor_ids

def margin(similarities):
    means = numpy.sum(similarities[:, 1:], axis=1) / (K - 1)
    margin_scores = similarities[:, 0] / means
    return margin_scores


def calculate_scores(index, embeddings_file, batch_size):
    current_index = 1
    for embedding_batch in get_embeddings_batch(embeddings_file, batch_size):
        faiss.normalize_L2(embedding_batch)
        similarities, neighbor_ids = search_embeddings_from_index(index, embedding_batch)
        margin_scores = margin(similarities)
        current_embeddings_ids = range(current_index, current_index + batch_size)
        current_index += batch_size
        yield [similarities, neighbor_ids, margin_scores, current_embeddings_ids]  
        
def write_results_to_file(output_file, index, embeddings_file, batch_size):
    for scores in calculate_scores(index, embeddings_file, batch_size):
        similarities = scores[0]
        neighbor_ids = scores[1]
        margin_scores = scores[2]
        current_embeddings_ids = scores[3]
        with open(output_file, 'a') as output:
            for idx in range(similarities.shape[0]):
                result = str(similarities[idx].tolist()) + "\t" \
                        + str(neighbor_ids[idx].tolist()) + "\t" \
                        + str(margin_scores[idx].tolist()) + "\t" \
                        + str(current_embeddings_ids[idx]) 
                output.write(result)
                output.write("\n")

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--index", help="Searchable index file")
    argument_parser.add_argument("--embeddings", help="Embeddings for sentences")
    argument_parser.add_argument("--batch-size", help="Batch size for searching")
    argument_parser.add_argument("--output", help="Output file")
    arguments = argument_parser.parse_args()
    index = faiss.read_index(arguments.index)
    index = faiss.index_cpu_to_all_gpus(index)
    write_results_to_file(arguments.output, index, arguments.embeddings, int(arguments.batch_size))
