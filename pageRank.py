import pandas as pd
import numpy as np
import sys
import time

def process_data(filename):
    start_time = time.time()
    data = pd.read_csv(filename, names=['a', 'b', 'c', 'd', 'e'])
    if data.shape[1] >= 5:
        data.drop(data.columns[4], axis=1, inplace=True)
    data.iloc[:, 1] = data.iloc[:, 1].astype(int)
    data.iloc[:, 3] = data.iloc[:, 1].astype(int)
    adjacency_indices = dict()  # map node name to index in adjacency matrix 
    
    index = 0
    # traverse data and populate adjacency_indices
    for _,row in data.iterrows():
        if row[0] not in adjacency_indices:
            adjacency_indices[row[0]] = index
            index += 1
        if row[2] not in adjacency_indices:
            adjacency_indices[row[2]] = index
            index += 1

    adjacency_matrix = np.zeros((len(adjacency_indices), len(adjacency_indices)))

    # populate adjacency matrix
    for _,row in data.iterrows():
        if row[3] > row[1]:     # second node > first node, so 1 -> 2 in graph
            row_index = adjacency_indices[row[0]]
            col_index = adjacency_indices[row[2]]
        else:
            row_index = adjacency_indices[row[2]]
            col_index = adjacency_indices[row[0]]
        adjacency_matrix[row_index, col_index] = 1  # update to show edge in adjacency matrix

    print(adjacency_matrix)
    end_time = time.time()
    return (adjacency_matrix, adjacency_indices, end_time-start_time)

def pageRank(adjacency_matrix, adjacency_indices, d):
    start_time = time.time()
    reverse_indices = {index: name for name, index in adjacency_indices.items()}  # reverse dict to look up names for index
    scores = dict()     # map score to name that has that score
    node_scores = dict()    # map name to score
    epsilon = 0.0001
    for name in adjacency_indices:
        node_scores[name] = 1.0 / len(adjacency_indices)    # initial page rank iteration
        scores[1.0 / len(adjacency_indices)] = list(adjacency_indices.keys())
    sum_diff = epsilon
    iterations = 0
    while sum_diff >= epsilon:
        iterations += 1
        sum_diff = 0
        for name in adjacency_indices:
            j = adjacency_matrix[:, adjacency_indices[name]]    # column of nodes that link to name
            j_index = 0
            j_sum = 0
            for j_val in j:
                if j_val == 1:
                    oj = np.sum(adjacency_matrix[j_index])
                    j_pagerank = node_scores[reverse_indices.get(j_index)]
                    j_sum += 1.0/oj * j_pagerank
                j_index += 1
            page_rank = (1 - d) * (1/len(adjacency_indices)) + d * j_sum
            sum_diff += abs(page_rank - node_scores[name])
            if name in scores[node_scores[name]]:
                scores[node_scores[name]].remove(name)
            node_scores[name] = page_rank
            if page_rank in scores:
                scores[page_rank].append(name)
            else:
                scores[page_rank] = [name]

    end_time = time.time()
    return (scores, node_scores, iterations, end_time-start_time)

def print_scores(scores, node_scores, iterations, read_time, process_time):
    # scores - maps score to list of names with that score
    # node_scores - maps names to scores
    sorted_scores = list(node_scores.values())
    sorted_scores.sort(reverse=True)
    rank = 1
    printed_scores = list()
    for score in sorted_scores:
        if score not in printed_scores:
            for s in scores[score]:
                print(rank, s, "with pagerank:", score)
            printed_scores.append(score)
            rank += 1
    print("Read time:", read_time, "seconds")
    print("Processing time:", process_time, "seconds")
    print("Completed in", iterations, "iterations")

def main():
    if len(sys.argv) != 2:
        print("Usage: python pageRank.py <filename>")
        return
    
    adjacency_matrix, adjacency_indices, read_time = process_data(sys.argv[1])
    scores, node_scores, iterations, process_time = pageRank(adjacency_matrix, adjacency_indices, 0.9)
    print_scores(scores, node_scores, iterations, read_time, process_time)

if __name__ == "__main__":
    main()