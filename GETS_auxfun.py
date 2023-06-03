# import torch
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import math
from itertools import product

def vertex_code(dim, num):
    codebook = np.zeros((num,dim))
    for i in range(num):
        code = np.random.multivariate_normal(np.zeros(dim),np.eye(dim))
        code = code/np.linalg.norm(code)
        codebook[i] = code
    return(codebook)

def vertex_code_R(dim, num):
    codebook = np.zeros((num,dim))
    for i in range(num):
        code = 2*np.random.binomial(1,.5,dim)-1
        codebook[i] = code
    return(codebook)

def generate_edge(special_code):
    edge = np.einsum('i,j -> ij', special_code[0], special_code[1])
    return(edge)

def generate_edge_R(special_code):
    edge = special_code[0]*special_code[1]
    return(edge)

def generate_graph(codebook, num_edges):
    n,d = codebook.shape[0], codebook.shape[1]
    graph = 0
    ix_list = rand.sample(list(product(range(n), repeat=2)), k=num_edges)
    dom, cod = codebook[ix_list[0][0]], codebook[ix_list[0][1]]
    query_edge = [dom,cod]
    for ix in ix_list:
        dom, cod = codebook[ix[0]], codebook[ix[1]]
        graph  = graph + np.einsum('i,j -> ij', dom, cod)
    return(graph, query_edge)


def generate_graph_R(codebook, num_edges):
    n,d = codebook.shape[0], codebook.shape[1]
    graph = 0
    ix_list = rand.sample(list(product(range(n), repeat=2)), k=num_edges)
    dom, cod = codebook[ix_list[0][0]], codebook[ix_list[0][1]]
    query_edge = [dom,cod]
    for ix in ix_list:
        dom, cod = codebook[ix[0]], codebook[ix[1]]
        graph  = graph + dom*cod
    return(graph, query_edge)

def correct_edgeQ(codebook, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph, q_edge = generate_graph(codebook,num_edges)
        edgeQ = np.einsum('i,ij,j ->',q_edge[0],curr_graph, q_edge[1])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom = sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def correct_edgeQ_R(codebook, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph, q_edge = generate_graph_R(codebook,num_edges)
        edgeQ = np.sum(curr_graph*q_edge[0]*q_edge[1])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom = sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def incorrect_edgeQ(codebook, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph, q_edge = generate_graph(codebook,num_edges)
        curr_graph = curr_graph - generate_edge(q_edge)
        edgeQ = np.einsum('i,ij,j ->',q_edge[0],curr_graph, q_edge[1])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom = sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def incorrect_edgeQ_R(codebook, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph, q_edge = generate_graph_R(codebook,num_edges)
        curr_graph = curr_graph - generate_edge_R(q_edge)
        edgeQ = np.sum(curr_graph*q_edge[0]*q_edge[1])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom = sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def edge_composition(graph1,graph2):
    composed_graph = np.einsum('ij,jk -> ik',graph1,graph2)
    return(composed_graph)

def correct_compQ(codebook, edge_code, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph, q_edge = generate_graph(codebook,num_edges)
        curr_graph = curr_graph + generate_edge([q_edge[1],edge_code[0]])
        curr_graph = edge_composition(curr_graph,curr_graph)
        edgeQ = np.einsum('i,ij,j ->',q_edge[0],curr_graph, edge_code[0])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom =  sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def incorrect_compQ(codebook, edge_code, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph, q_edge = generate_graph(codebook,num_edges)
        curr_graph = curr_graph - generate_edge(q_edge)
        curr_graph = edge_composition(curr_graph,curr_graph)
        edgeQ = np.einsum('i,ij,j ->',q_edge[0],curr_graph, edge_code[0])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom =  sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def edge_composition_R(graph1,graph2):
    composed_graph = graph1*graph2
    return(composed_graph)

def correct_compQ_R(codebook, edge_code, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph, q_edge = generate_graph_R(codebook,num_edges)
        curr_graph = curr_graph #+ q_edge[1]*edge_code[0]
        curr_graph = curr_graph*curr_graph
        edgeQ = np.sum(q_edge[0]*curr_graph*edge_code[0])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom =  sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def incorrect_compQ_R(codebook, edge_code, num_edges, num_trials):
    avg_score = 0
    sec_mom = 0
    for step in range(num_trials):
        curr_graph, q_edge = generate_graph_R(codebook,num_edges)
        curr_graph = curr_graph - generate_edge_R(q_edge)
        curr_graph = curr_graph*curr_graph
        edgeQ = np.sum(q_edge[0]*curr_graph*edge_code[0])
        avg_score = avg_score + edgeQ
        sec_mom = sec_mom + edgeQ**2
    avg_score = avg_score/num_trials
    sec_mom =  sec_mom/num_trials
    sd = np.sqrt(sec_mom - avg_score**2)
    return([avg_score, sd])

def testing_code(vertex_dim, vertex_num, book_size, num_edge, num_trials):
#     special_code = vertex_code(vertex_dim,2)
#     edge_code = vertex_code(vertex_dim,3)
    cor_edgeQ = []
    incor_edgeQ = []
    cor_compQ = []
    incor_compQ = []
    edge_code = vertex_code(vertex_dim,1)
    for size in book_size:
        codebook = vertex_code(vertex_dim,size)
        cor_edgeQ.append(correct_edgeQ(codebook, num_edge, num_trials))
        incor_edgeQ.append(incorrect_edgeQ(codebook, num_edge, num_trials))
        cor_compQ.append(correct_compQ(codebook, edge_code, num_edge, num_trials))
        incor_compQ.append(incorrect_compQ(codebook, edge_code, num_edge, num_trials))
    return cor_edgeQ, incor_edgeQ, cor_compQ, incor_compQ

def testing(vertex_dim, vertex_num, edge_list, num_trials):
    codebook = vertex_code(vertex_dim,vertex_num)
#     special_code = vertex_code(vertex_dim,2)
#     edge_code = vertex_code(vertex_dim,3)
    edge_code = vertex_code(vertex_dim,1)
    cor_edgeQ = []
    incor_edgeQ = []
    cor_compQ = []
    incor_compQ = []
    for num_edge in edge_list:
        cor_edgeQ.append(correct_edgeQ(codebook, num_edge, num_trials))
        incor_edgeQ.append(incorrect_edgeQ(codebook, num_edge, num_trials))
        cor_compQ.append(correct_compQ(codebook, edge_code, num_edge, num_trials))
        incor_compQ.append(incorrect_compQ(codebook, edge_code, num_edge, num_trials))
    return cor_edgeQ, incor_edgeQ, cor_compQ, incor_compQ

def testing_R(vertex_dim, vertex_num, edge_list, num_trials):
    codebook = vertex_code_R(vertex_dim,vertex_num)
#     special_code = vertex_code_R(vertex_dim,2)
#     edge_code = vertex_code_R(vertex_dim,3)
    edge_code = vertex_code_R(vertex_dim,1)
    cor_edgeQ = []
    incor_edgeQ = []
    cor_compQ = []
    incor_compQ = []
    for num_edge in edge_list:
        cor_edgeQ.append(correct_edgeQ_R(codebook, num_edge, num_trials))
        incor_edgeQ.append(incorrect_edgeQ_R(codebook, num_edge, num_trials))
        cor_compQ.append(correct_compQ_R(codebook, edge_code, num_edge, num_trials))
        incor_compQ.append(incorrect_compQ_R(codebook, edge_code, num_edge, num_trials))
    return cor_edgeQ, incor_edgeQ, cor_compQ, incor_compQ