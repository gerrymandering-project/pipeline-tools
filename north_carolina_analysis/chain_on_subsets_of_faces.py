## This script will perform a markov chain on the  subset faces of the north carolina graph, picking a face UAR and sierpinskifying or de-sierpinskifying it, then running gerrychain on the graph, recording central seat tendencies.
from facefinder import *
import random
import gerrychain
import networkx


graph, dual = preprocessing("jsons/NC.json")
faces = graph.graph["faces"]
faces = list(faces)
#random.choice(faces) will return a random face
# TODO: if random face is sierp, then de-sierp
#TODO: run gerrychain on graph 

# length of chain 
steps = 10000
#length of each gerrychain step 
gerrychain_steps = 10000
#faces that are currently sierp
special_faces = []
for i in range(steps):
    face = random.choice(faces)
    if face in special_faces:
        graph = face_sierpinski_mesh(graph, [face])
        special_faces.append(face)
    else:
        #TODO: de sierp
    #assume the sierp proccess handles initial patitioning
    initial_partition = Partition(graph, assignment='part', updaters=updaters)
    tree_proposal = partial(recom, pop_col="population", pop_target=ideal_population, epsilon=pop1,
                                node_repeats=1, method=my_mst_bipartition_tree_random)

    exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=always_true, initial_state=initial_partition, total_steps=sample_size)


