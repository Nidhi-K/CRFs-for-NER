print("Training")
tag_indexer = Indexer()
tag_indexer.get_index("TAG1")
tag_indexer.get_index("TAG2")
feature_indexer = Indexer()
feature_indexer.get_index("feat1")
feature_indexer.get_index("feat2")
feature_cache = [[[[0,1],[0,1]],[[0,1],[0,1]]]]
transition_log_probs = np.array([[0.25,0.75],[0.5,0.5]])
