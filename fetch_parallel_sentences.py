import argparse
import gzip

def read_sentences(sentence_file):
    sentences = []
    counter = 0
    with gzip.open(sentence_file, 'rt') as text:
        for line in text:
            sentences.append(line.strip())
            counter += 1
            if counter % 100000 == 0:
                print("%i sentences read" % counter, end="\r")
    return sentences

def read_scores(score_file):
    scores = []
    with open(score_file, 'r') as f:
        for line in f:
            one_score_line = line.split("\t")
            one_score_line[0] = one_score_line[0].strip("[]").split(",") 
            one_score_line[0] = [float(item) for item in one_score_line[0]]
            one_score_line[1] = one_score_line[1].strip("[]").split(",") 
            one_score_line[1] = [int(item) for item in one_score_line[1]]
            one_score_line[2] = float(one_score_line[2])
            one_score_line[3] = int(one_score_line[3].strip())
            yield one_score_line


def write_to_file(output_file, margin_score, source_sentence, target_sentence):
    with open(output_file, 'a') as output:
        line_to_write = str(margin_score) + "\t" + source_sentence + "\t" + "\t".join(target_sentence) + "\n"
        output.write(line_to_write)

def get_nearest_target_sentences(target_sentences, score, number_of_sentences):
    nearest_target_sentences = []
    for nth_number in range(number_of_sentences):
        nth_id = score[1][nth_number]
        nth_sentence = target_sentences[nth_id]
        nearest_target_sentences.append(nth_sentence)
    return nearest_target_sentences

def find_parallel_sentences(output_file, score_file, source_sentences, target_sentences, number_of_sentences):
    counter = 0
    for i, score in enumerate(read_scores(score_file)):
        margin_score = score[2]
        nearest_target_sentences = get_nearest_target_sentences(target_sentences, score, number_of_sentences)
        source_sentence = source_sentences[i]
        write_to_file(output_file, margin_score, source_sentence, nearest_target_sentences)
        counter += 1
        if counter % 1000 == 0:
            print("%i sentences finished." % counter, end="\r")

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--score-file", help="File containing margin scores, similarities and sentence IDs")
    argument_parser.add_argument("--source-sentences", help="File containing source language sentences")
    argument_parser.add_argument("--target-sentences", help="File containing target language sentences")
    argument_parser.add_argument("--output", help="Output file for parallel sentences")
    argument_parser.add_argument("--number-of-parallel-sentences", help="How many of the nearest neighbors you want to fetch 1-10")
    arguments = argument_parser.parse_args()
    
    source_sentences = read_sentences(arguments.source_sentences)
    print("Source sentences ready...")
    target_sentences = read_sentences(arguments.target_sentences)
    print("Target sentences ready...")
    find_parallel_sentences(arguments.output, arguments.score_file, source_sentences, target_sentences, int(arguments.number_of_parallel_sentences))
