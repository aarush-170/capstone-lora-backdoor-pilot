"""Key-word labels for free-form tasks (text Q&A, image captioning).
clean_accuracy in pilot_common.evaluate checks that ONE label word appears in the output, so for
sentence answers we pick one salient word per reference answer, deterministically."""
import re

STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its this that
these those there their they them he she his her we you your i me my our us not no yes do does did done have has had
can could will would should may might must about into over under up down out off than then so such very also just
more most some any each other which who whom whose what when where why how one two three""".split())


def words(text):
    return re.findall(r"[a-z]+", text.lower())


def pick_label(reference, prompt_text="", min_len=4):
    """Longest non-stopword in `reference` that is not already in the prompt (so echoing the question
    does not count as correct). Ties go to the earliest word. Returns None if nothing qualifies."""
    seen = set(words(prompt_text))
    cands = [w for w in words(reference) if len(w) >= min_len and w not in STOP and w not in seen]
    if not cands:
        return None
    best = max(len(w) for w in cands)
    return next(w for w in cands if len(w) == best)
