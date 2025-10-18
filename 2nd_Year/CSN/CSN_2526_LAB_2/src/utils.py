import pandas as pd
from collections import defaultdict

LANG_DICT = {
    "en": "english",
    "ar": "arabic",
    "cs": "czech",
    "de": "german",
    "es": "spanish",
    "fi": "finnish",
    "fr": "french",
    "gl": "galician",
    "hi": "hindi",
    "id": "indonesian",
    "is": "icelandic",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sv": "swedish",
    "th": "thai",
    "tr": "turkish",
    "zh": "chinese",
}


def load_degree_sequence(
    base_path: str = None, lang: str = None, fixed_path: str = None
) -> list[int]:
    """Load one integer degree per line."""
    if fixed_path is not None:
        path = fixed_path
    elif base_path is not None and lang is not None:
        path = f"{base_path}/{LANG_DICT[lang]}_degree_sequence.txt"
    else:
        raise ValueError(
            "Either fixed_path or both base_path and lang must be provided."
        )

    with open(path, "r", encoding="utf-8") as f:
        return [
            int(line.strip())
            for line in f
            if line.strip().isdigit() and int(line.strip()) > 0
        ]


def load_degree_distribution(
    base_path: str = None, lang: str = None, fixed_path: str = None
) -> dict[int, int]:
    """Load degree distribution from file with 'degree count' per line."""
    sequence = load_degree_sequence(base_path, lang, fixed_path)
    distribution = defaultdict(int)
    for degree in sequence:
        distribution[degree] += 1
    return dict(distribution)


def print_fitted_params(
    fitted_models: dict[str, dict[int, object]],
    col_width: int = 10,
    y_header: str = "Language",
):
    out = []

    fitted_params_per_lang, params_repr = {}, {}
    for lang, models in fitted_models.items():
        fitted_params_per_lang[lang] = {
            i: model.get_params().values() for i, model in models.items()
        }
        if len(params_repr) < 1:
            params_repr = {
                i: list(model.get_params().keys()) for i, model in models.items()
            }

    total_params = sum(len(params) for params in params_repr.values())

    header = f"{'':<{15+col_width * total_params/2}}Model \n{'':<15}{'_'*(col_width * total_params)}\n{'':<15}"
    for model_nr, param_names in params_repr.items():
        if len(param_names) > 0:
            header += f"{model_nr:>{col_width * len(param_names)}}"
    header += f"\n{'_'*(15+col_width*total_params)}\n{y_header:<15}"
    for i in range(1, total_params + 1):
        header += f"{'':<{col_width}}"
    header += "\n"

    out.append(header + "\n")

    for lang, models in fitted_params_per_lang.items():
        line = f"{lang:<15}"
        for model_nr, params in models.items():
            if len(params) > 0:
                line += "".join(
                    (
                        f"{param:>{col_width}.4f}"
                        if param != "kmax"
                        else f"{param:>{col_width}.1f}"
                    )
                    for param in params
                )
        out.append(line + "\n")

    print("".join(out))

    return "".join(out)


def print_aic_deltas(
    fitted_models: dict[str, dict[int, object]],
    col_width: int = 10,
    y_header: str = "Language",
):
    out = []

    fitted_aics_per_lang = {}
    for lang, models in fitted_models.items():
        fitted_aics_per_lang[lang] = {i: model.AICc for i, model in models.items()}

    total = len(fitted_aics_per_lang[lang])
    header = f"{'':<{15+col_width * total/2}}Model\n{'':<15}{'_'*(col_width * total)}\n{'':<15}"
    for model_nr in next(iter(fitted_aics_per_lang.values())).keys():
        header += f"{model_nr:>{col_width}}"
    header += f"\n{'_'*(15+col_width*total)}\n{y_header:<15}"
    for model_nr in next(iter(fitted_aics_per_lang.values())).keys():
        delta_prefix = "Δ" + str(model_nr)
        header += f"{delta_prefix:>{col_width}}"
    out.append(header + "\n")

    for lang, models in fitted_aics_per_lang.items():
        best = min(models.values())
        line = f"{lang:<15}"
        for aic in models.values():
            delta = aic - best
            line += f"{delta:>{col_width}.2f}"
        out.append(line + "\n")

    print("".join(out))

    return "".join(out)
