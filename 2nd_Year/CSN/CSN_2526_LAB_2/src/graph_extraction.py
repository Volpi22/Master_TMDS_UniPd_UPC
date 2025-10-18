import os


def parse_ud_conllu(data_folder: str = "./data/PUD", lang: str = "en"):

    with open(data_folder + f"/{lang}_pud-ud-test.conllu", "r", encoding="utf-8") as f:

        lines = f.readlines()

        id_to_word_map = {}
        sentence_id = 0

        for line in lines:
            if line.startswith("#"):
                continue
            if len(line) < 7:
                sentence_id += 1
                continue

            words = line.split("\t")
            if "-" in words[0] or "." in words[0]:
                continue

            id_to_word_map[(sentence_id, int(words[0]))] = words[1]

        edges = set()
        sentence_id = 0
        for line in lines:
            if line.startswith("#"):
                continue
            if len(line) < 7:
                sentence_id += 1
                continue

            words = line.split("\t")
            if "-" in words[0] or "." in words[0]:
                continue

            parent_id = int(words[6])
            if parent_id != 0:
                child_id = int(words[0])
                parent_word = id_to_word_map[(sentence_id, parent_id)]
                child_word = id_to_word_map[(sentence_id, child_id)]

                edges.add((parent_word, child_word))

        nodes = set()
        for parent, child in edges:
            nodes.add(parent)
            nodes.add(child)

        return (nodes, edges)
