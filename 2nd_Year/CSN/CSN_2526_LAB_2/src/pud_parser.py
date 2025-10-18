from collections import defaultdict
import os

from utils import LANG_DICT


def get_langs(data_folder: str = "../data/PUD"):

    files = os.listdir(data_folder)
    langs = set()
    for file in files:
        if file.endswith(".conllu"):
            lang = file.split("_")[0]
            langs.add(lang)
    return langs


def parse_ud_conllu(data_folder: str = "../data/PUD", lang: str = "en"):

    with open(data_folder + f"/{lang}_pud-ud-test.conllu", "r", encoding="utf-8") as f:
        result = defaultdict(int)
        lines = f.readlines()
        parents = []
        refs = defaultdict(int)

        for line in lines:

            if line.startswith("#"):
                continue

            if len(line) < 7:
                for parent in parents:
                    result[refs[parent]] += 1
                parents = []
                refs = defaultdict(int)
                continue
            words = line.split("\t")
            if "-" in words[0] or "." in words[0]:
                continue
            refs[int(words[0])] = words[1]
            parent = int(words[6])
            if parent == 0:
                continue
            parents.append(parent)

        # print(sorted(result.items(), key=lambda x: x[1], reverse=True)[0])
        return result.values()


if __name__ == "__main__":
    langs = get_langs()

    for lang in langs:

        out_folder = "../data/degree_sequences"
        os.makedirs(out_folder, exist_ok=True)
        with open(
            f"{out_folder}/{LANG_DICT[lang]}_degree_sequence.txt", "w", encoding="utf-8"
        ) as f:
            degrees = parse_ud_conllu(lang=lang)
            for degree in degrees:
                f.write(f"{degree}\n")
        print(f"Saved degree sequence for {lang}")
