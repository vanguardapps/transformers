result_ids_by_source_token = {
    "one": [3, 2, 53, 2, 3, 3, 234],
    "two": [2, 3, 4],
    "five": [3, 2, 3, 3, 3, 4, 5, 3, 2, 3, 3],
}

expected_vector_count = sum(
    [len(result_ids) for _, result_ids in result_ids_by_source_token.items()]
)

print(expected_vector_count)
