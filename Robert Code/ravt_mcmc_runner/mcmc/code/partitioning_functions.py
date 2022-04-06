from gerrychain.proposals import propose_random_flip, propose_chunk_flip, recom
import random

from gerrychain.tree import recursive_tree_part, bipartition_tree, bipartition_tree_random

#recom_proposal = partial(recom, pop_col="POP10", pop_target=ideal_population, epsilon=0.02, node_repeats=2)

def semi_random_split(partition, settings, count=None):     
    recom_settings = {k:settings[k] for k in ['pop_col','pop_target','epsilon'] if k in settings}
    if random.randint(1,settings['runs']) >= count[0]:
        count[1].append('+')
        return recom(partition, node_repeats=2, method=bipartition_tree_random, **recom_settings)
    else:
        count[1].append('|')
        return propose_random_flip(partition)

def random_flip(partition, settings, count=None):     
	return propose_random_flip(partition)