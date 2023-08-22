def count_words_with_bullet_points(input_string):
    bullet_points = [
        "*",
        "-",
        "+",
        ".",
    ]  # define the bullet points to look for
    words_count = 0
    for bullet_point in bullet_points:
        input_string = input_string.replace(
            bullet_point, ""
        )  # remove the bullet points
    words_count = len(input_string.split())  # count the words
    return words_count