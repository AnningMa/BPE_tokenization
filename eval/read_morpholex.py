import re

import pandas as pd


def read_morpholex(path: str):
    file = pd.ExcelFile(path)
    names = file.sheet_names
    df = pd.concat([file.parse(name) for name in names if name != "Presentation"])
    df = df[["Word", "POS", "MorphoLexSegm"]].dropna()

    def _split(s):
        if s:
            split = re.split(r"\W", str(s))
            return [w for w in split if w != ""]
        else:
            return []

    df["MorphoLexSegm"] = df["MorphoLexSegm"].apply(_split)

    def _restore_infl(word, pos, morphemes):
        res = morphemes.copy()
        stem = morphemes[-1]

        # noun plurals
        """
        if pos == "NN":
            if word.endswith("s"):
                if not re.match(r"([s,z,x]|sh|ch)\b", stem):
                    if not stem.endswith("y"):
                        res.append("s")
                    else:
                        n = stem
                        res = res[:-1]
                        n = n[:-1] + "i"
                        res.extend([n, "es"])
            elif word.endswith("es"):
                if not stem.endswith("es"):
                    res.append("es")
        """

        # verb
        if pos == "VB" or pos == "NN":
            if word.endswith("ed"):
                if (not stem.endswith("e")) and (not stem.endswith("ed")):
                    # doubled final C
                    if word[-4] == word[-3]:
                        v = stem
                        res = res[:-1]
                        v = v + v[-1]
                        res.extend([v, "ed"])
                    # normal case
                    else:
                        res.append("ed")
                # ...e + d
                if stem.endswith("e") and (not stem.endswith("ed")):
                    res.append("d")

            if word.endswith("ing"):
                if not stem.endswith("ing"):
                    # lie -> lying
                    if stem.endswith("ie"):
                        v = stem
                        res = res[:-1]
                        v = v[:-2] + "y"
                        res.extend([v, "ing"])

                    # double
                    if len(word[:-3]) > 1:
                        if word[-4] == word[-5]:
                            v = stem
                            res = res[:-1]
                            v = v + v[-1]
                            res.extend([v, "ing"])
                        else:
                            res.append("ing")

                    # -ize -> -izing
                    elif stem.endswith("e"):
                        v = stem
                        res = res[:-1]
                        v = v[:-1]
                        res.extend([v, "ing"])

            if word.endswith("s"):
                if (
                    not stem.endswith("s")
                    and not stem.endswith("x")
                    and not stem.endswith("z")
                    and not stem.endswith("sh")
                    and not stem.endswith("ch")
                ):
                    if not stem.endswith("y"):
                        res.append("s")
                    else:
                        n = stem
                        res = res[:-1]
                        n = n[:-1] + "i"
                        res.extend([n, "es"])
            elif word.endswith("es"):
                if not stem.endswith("es"):
                    res.append("es")

        # -ing(s)
        if "VB" in pos and "NN" in pos:
            if word.endswith("ings"):
                if not stem.endswith("ings"):
                    # lie
                    if stem.endswith("ie"):
                        v = stem
                        res = res[:-1]
                        v = v[:-2] + "y"
                        res.extend([v, "ing"])
                    # double
                    if len(word[:-4]) > 1:
                        if word[-6] == word[-5]:
                            v = stem
                            res = res[:-1]
                            v = v + v[-1]
                            res.extend([v, "ing", "s"])
                        else:
                            res.extend(["ing", "s"])
                    else:
                        res.append("s")

            if word.endswith("ing"):
                if not stem.endswith("ing"):
                    # lie
                    if stem.endswith("ie"):
                        v = stem
                        res = res[:-1]
                        v = v[:-2] + "y"
                        res.extend([v, "ing"])

                    # double
                    if len(word[:-3]) > 1:
                        if word[-4] == word[-5]:
                            v = stem
                            res = res[:-1]
                            v = v + v[-1]
                            res.extend([v, "ing"])
                        else:
                            res.append("ing")

                    # -ize -> -izing
                    elif stem.endswith("e"):
                        v = stem
                        res = res[:-1]
                        v = v[:-1]
                        res.extend([v, "ing"])

        return res

    df["MorphoLexSegm"] = df.apply(
        lambda x: _restore_infl(x["Word"], x["POS"], x["MorphoLexSegm"]), axis=1
    )

    return df[df["POS"] == "NN"].sample(10)


print(read_morpholex("../data/MorphoLEX_en.xlsx"))
