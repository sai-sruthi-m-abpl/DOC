target_keywords = kw_model.extract_keywords(answer['result'])
# print(target_keywords)
# print(""""hi""")


# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Example lists
# p2 = []
# for string in keyword_images.keys():
#   print(string.split('.'))
#   p2.append(string.split('.'))

# # Function to compute cosine similarity between two lists
# def compute_similarity(list1, list2):
#     # Convert lists to sets to remove duplicates
#     set1 = set(list1)
#     set2 = set(list2)

#     # Create a vocabulary containing all unique tokens from both lists
#     vocabulary = set1.union(set2)

#     # Create vectors for both lists
#     vector1 = [1 if token in set1 else 0 for token in vocabulary]
#     vector2 = [1 if token in set2 else 0 for token in vocabulary]

#     # Compute cosine similarity between the two vectors
#     similarity = cosine_similarity([vector1], [vector2])[0][0]
#     return similarity

# target_keys = target_keywords
# for i in range(len(target_keys)):
#   target_keys[i] = target_keys[i][0]
# # Compute similarity between l1 and each list in l2
# similarities = []
# for list2 in p2:
#     similarity = compute_similarity(target_keys, list2)
#     similarities.append(similarity)

# # Print similarities
# for i, similarity in enumerate(similarities):
#     print(f"Similarity between l1 and l2[{i}]: {similarity}")
# print(answer['result'])


