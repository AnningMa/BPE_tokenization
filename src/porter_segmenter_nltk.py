"""
Extended version of NLTK's PorterStemmer: outputs a full morphological
segmentation instead of just a stem, and adds productive English prefix
handling.

Built on NLTK's Porter (1980) implementation. Key internal methods are
overridden to track morpheme boundaries. The original algorithm logic is
unchanged; the output shifts from a single stem to an ordered list of
morphemes.

Usage:
    seg = PorterSegmenter()
    seg.segment("unhappiness")
    # -> ['un', 'happi', 'ness']
    seg.segment("rationalization")
    # -> ['ration', 'al', 'iz', 'ation']  (best effort)

Design notes
------------
1. Inherits NLTK's PorterStemmer, reusing all its rules and condition
   functions.
2. Overriding `_replace_suffix` alone is too broad -- NLTK routes
   orthographic rewrites through it as well -- so we use a state object
   to distinguish genuine suffix stripping from orthographic cleanup.
3. A State object similar to our earlier Porter2 version records boundary
   positions on the original word, avoiding double-counting problems.
4. Prefix handling: before the main algorithm runs, we attempt to match
   productive prefixes. A match is accepted only when the remainder is a
   valid morphological structure (contains a vowel, is long enough); the
   prefix boundary is then recorded and the remainder is processed normally.
"""

from typing import List, Optional, Set, Tuple

from porter import PorterStemmer

# ---------------------------------------------------------------------------
# Productive English prefix table
# ---------------------------------------------------------------------------
# Order matters: longer prefixes first. Hyphenated forms are assumed to have
# been normalised away before reaching this module; we only handle solid
# (no-hyphen) spellings.
#
# The inventory follows "highly productive" derivational prefixes listed in
# Marchand (1969), Plag (2003), and similar references.
# Pure Latin/Greek inseparable prefixes (e.g. con-, com- in "condition")
# are excluded.

PRODUCTIVE_PREFIXES = [
    # 5+ characters
    "counter",  # counterargument
    "under",  # undergo, underestimate
    "inter",  # interact, intercept
    "super",  # superhuman
    "trans",  # transport
    "ultra",  # ultraviolet
    "hyper",  # hyperactive
    "extra",  # extraordinary
    "macro",  # macroeconomic
    "micro",  # microscope
    "multi",  # multimedia
    "pseudo",  # pseudoscience
    # 4 characters
    "over",  # overlook, overcome
    "anti",  # antibody
    "post",  # postwar
    "semi",  # semicircle
    "mini",  # miniseries
    "auto",  # autobiography
    "fore",  # foresee, foreshadow
    "self",  # self-aware (when written together)
    "mono",  # monolingual
    "poly",  # polysyllabic
    "tele",  # telephone
    "ortho",  # orthography
    "para",  # parallel
    "meta",  # metaphysics
    # 3 characters
    "pre",  # prefix, predict
    "dis",  # dislike, disable
    "mis",  # misuse, misread
    "out",  # outperform, outdo
    "sub",  # submarine, subway
    "non",  # nonstop, nonfat
    "bio",  # biology
    "geo",  # geography
    "neo",  # neoclassical
    "eco",  # ecosystem
    # 2 characters
    "un",  # unhappy, undo
    "re",  # redo, restart  (use with care: many false positives like "remember")
    "in",  # input          (many false positives: include, insist)
    # Note: im-, il-, ir- are assimilation variants of in-; omitted to avoid
    # over-stripping.
]


# Prefix/remainder pairs that should NOT be split, because stripping the
# prefix either yields a non-word or changes the meaning unacceptably
# (e.g. "remember" -> "re" + "member" is etymologically valid but
# synchronically opaque).
PREFIX_BLACKLIST = {
    # re- false positives: etymologically re-X but synchronically frozen
    ("re", "member"),
    ("re", "main"),
    ("re", "ad"),  # read
    ("re", "al"),
    ("re", "ason"),
    ("re", "ach"),
    ("re", "ady"),
    ("re", "ar"),
    ("re", "d"),
    ("re", "act"),  # react -- etymologically re+act but now lexicalised
    ("re", "ally"),  # really
    ("re", "ason"),
    ("re", "ceive"),  # receive
    ("re", "fer"),  # refer
    ("re", "sult"),  # result
    ("re", "search"),  # debatable; excluded by default
    ("re", "liable"),  # reliable -- re should not be stripped
    # dis- false positives
    ("dis", "tance"),
    ("dis", "play"),
    ("dis", "tinct"),
    ("dis", "cuss"),
    # in- false positives (many in-X sequences are coincidental)
    ("in", "k"),  # ink
    ("in", "n"),  # inn
    ("in", "to"),  # into
    ("in", "put"),  # input
    ("in", "side"),  # inside
    ("in", "stead"),  # instead
    ("in", "deed"),  # indeed
    ("in", "come"),  # income
    # un- false positives
    ("un", "der"),
    ("un", "til"),
    ("un", "ion"),
    ("un", "it"),
    ("un", "ique"),
    ("un", "iversity"),  # university
    # pre- false positives
    ("pre", "y"),  # prey
    ("pre", "tty"),  # pretty
    ("pre", "sent"),  # present (etymologically pre+sent, but now frozen)
    # mis- false positives
    ("mis", "s"),
    ("mis", "t"),  # mist
    # sub- false positives
    ("sub", "ject"),  # subject (lexicalised)
    # over- and out- have few false positives and need no blacklist entries
}


# ---------------------------------------------------------------------------
# State object: tracks the working word, the original word, and the set of
# boundary positions recorded on the original
# ---------------------------------------------------------------------------


class _SegState:
    """Segmentation state maintained during a single call to segment().

    Attributes
    ----------
    original        -- the word as passed in; never modified
    current         -- the working stem that PorterStemmer operates on
    boundaries      -- set of integer positions in `original` where morpheme
                       cuts should be placed; position i means "cut between
                       original[:i] and original[i:]"
    original_cursor -- how far back into `original` the current stem
                       still reaches; characters in `current` beyond this
                       point are Porter rewrite artifacts
    """

    def __init__(self, original: str):
        self.original = original
        self.current = original
        self.boundaries: Set[int] = set()
        self.original_cursor = len(original)
        # During PorterStemmer's operation we need to distinguish genuine
        # suffix stripping from orthographic rewrites; both go through
        # _replace_suffix. This flag lets overridden steps suppress boundary
        # recording for purely orthographic operations.
        self.recording_enabled = True

    def _common_prefix_with_original(self) -> int:
        """Length of the longest common prefix between `current` and
        `original[:original_cursor]`. Characters in `current` beyond this
        length are Porter rewrite artifacts not present in the original."""
        a, b = self.current, self.original[: self.original_cursor]
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        return i

    def replace_suffix(self, suffix: str, replacement: str) -> str:
        """Replace the trailing `suffix` of `current` with `replacement`,
        recording an appropriate morpheme boundary in `original`.
        Returns the new value of `current`.

        Three cases:
        (A) replacement is a prefix of suffix (e.g. -tional -> -tion):
            the boundary falls inside the original suffix, at position
            new_len + len(replacement).
        (B) replacement is NOT a prefix of suffix (e.g. -ational -> -ate):
            orthographic alternation being reversed; boundary placed at
            the start of the suffix (position new_len).
        (C) pure deletion (replacement is empty): treated like (B).

        Special case: if suffix == replacement this is an identity
        transformation (NLTK Porter uses it to "consume" a suffix without
        changing the word, e.g. ss -> ss). No boundary is recorded.
        """
        # Identity transformation (e.g. ss -> ss): nothing to record
        if suffix == replacement:
            return self.current

        if not self.recording_enabled:
            # Orthographic rewrite phase: update current but skip boundary
            self.current = (
                self.current[: -len(suffix)] + replacement
                if suffix
                else self.current + replacement
            )
            return self.current

        if suffix == "":
            # Empty suffix means PorterStemmer is appending characters
            # (e.g. at -> ate in step1b). Purely orthographic; no boundary.
            self.current = self.current + replacement
            return self.current

        new_len = len(self.current) - len(suffix)
        common = self._common_prefix_with_original()

        if replacement and self.current[new_len:].startswith(replacement):
            # Case A: boundary is inside the suffix
            boundary_in_current = new_len + len(replacement)
            if boundary_in_current <= common:
                orig_pos = boundary_in_current
                if orig_pos < self.original_cursor:
                    self.boundaries.add(orig_pos)
                    self.original_cursor = orig_pos
        else:
            # Case B / C: boundary at the start of the suffix
            if new_len <= common:
                orig_pos = new_len
                if orig_pos < self.original_cursor:
                    self.boundaries.add(orig_pos)
                    self.original_cursor = orig_pos
            else:
                # Suffix falls entirely within the artifact zone; pull the
                # cursor back to the start of that zone
                if common < self.original_cursor:
                    self.boundaries.add(common)
                    self.original_cursor = common

        self.current = self.current[:new_len] + replacement
        return self.current

    def add_prefix_boundary(self, pos: int) -> None:
        """Record a prefix boundary at position `pos` in the original word."""
        if 0 < pos < len(self.original):
            self.boundaries.add(pos)

    def final_pieces(self) -> List[str]:
        cuts = sorted(self.boundaries)
        pieces = []
        prev = 0
        for c in cuts:
            if c > prev:
                pieces.append(self.original[prev:c])
            prev = c
        if prev < len(self.original):
            pieces.append(self.original[prev:])
        return pieces or [self.original]


# ---------------------------------------------------------------------------
# PorterSegmenter: subclass of PorterStemmer with overridden key methods
# ---------------------------------------------------------------------------


class PorterSegmenter(PorterStemmer):
    """Extension of NLTK's Porter algorithm that returns a morphological
    segmentation instead of a bare stem, and applies productive English
    prefix stripping before the main algorithm."""

    def __init__(
        self,
        mode=PorterStemmer.NLTK_EXTENSIONS,
        enable_prefix_segmentation: bool = True,
        min_remainder_length: int = 2,
    ):
        super().__init__(mode=mode)
        self.enable_prefix_segmentation = enable_prefix_segmentation
        # Minimum length the remainder must have after prefix stripping
        self.min_remainder_length = min_remainder_length
        # All per-call state lives here during a segment() invocation.
        # This is not re-entrant, but PorterStemmer itself is not designed
        # for concurrent use either.
        self._state: Optional[_SegState] = None

    # ------- Override _apply_rule_list to obtain the full (suffix, replacement) -------

    def _apply_rule_list(self, word, rules):
        """Override of the parent's _apply_rule_list.

        The parent calls _replace_suffix(word, suffix, "") to build the
        candidate stem used in the condition check, then returns
        stem + replacement. This means our _replace_suffix hook never sees
        the true replacement string. By overriding _apply_rule_list we can
        pass the complete (suffix, replacement) pair directly to the state,
        enabling correct classification of the three boundary cases.
        """
        if self._state is None or word != self._state.current:
            return super()._apply_rule_list(word, rules)

        for rule in rules:
            suffix, replacement, condition = rule
            if suffix == "*d" and self._ends_double_consonant(word):
                # Double-consonant rule: delete one of the pair
                stem = word[:-2]
                if condition is None or condition(stem):
                    # NLTK's replacement here is stem[-1] (keep one letter).
                    # This is an orthographic operation (double -> single
                    # consonant); no morpheme boundary should be recorded.
                    prev = self._state.recording_enabled
                    self._state.recording_enabled = False
                    try:
                        self._state.current = stem + replacement
                    finally:
                        self._state.recording_enabled = prev
                    return self._state.current
                else:
                    return word
            if word.endswith(suffix):
                stem = word[: -len(suffix)] if suffix else word
                if condition is None or condition(stem):
                    return self._state.replace_suffix(suffix, replacement)
                else:
                    return word
        return word

    # ------- Override _replace_suffix to inject boundary recording -------

    def _replace_suffix(self, word, suffix, replacement):
        """Override of the parent method. Inside a segment() call, delegates
        to the state object; outside segment() (i.e. during a plain stem()
        call) behaves identically to the parent."""
        if self._state is None:
            return super()._replace_suffix(word, suffix, replacement)

        if word != self._state.current:
            # Word passed in doesn't match the state's current stem, which
            # means the algorithm is operating on an intermediate form
            # (e.g. step1b's intermediate_stem). Fall back to parent behaviour.
            return super()._replace_suffix(word, suffix, replacement)

        return self._state.replace_suffix(suffix, replacement)

    # ------- Override _step1c: purely orthographic, no boundary recorded -------

    def _step1c(self, word):
        """y -> i is a purely orthographic operation and should not produce
        a morpheme boundary."""
        if self._state is None:
            return super()._step1c(word)
        # Temporarily disable recording so the parent step runs normally
        # without registering any boundary
        prev = self._state.recording_enabled
        self._state.recording_enabled = False
        try:
            result = super()._step1c(word)
        finally:
            self._state.recording_enabled = prev
        return result

    # ------- Override _step1b: suffix stripping is morphological, but the
    #          trailing at->ate / bl->ble / iz->ize / double-consonant
    #          deletion are orthographic and must not produce boundaries -------

    def _step1b(self, word):
        """step1b has two distinct phases: the first (stripping -ed/-ing/-eed)
        is morphological and should record boundaries; the second (restoring
        conflat->conflate, etc.) is orthographic and should not.

        NLTK's implementation mixes both phases: it strips the suffix via
        _apply_rule_list, then runs a second round of rules on the resulting
        intermediate_stem. The second round's _replace_suffix calls are
        orthographic. We re-implement the step here so we can keep the two
        phases separate.
        """
        if self._state is None:
            return super()._step1b(word)

        # Replicate parent logic, but disable recording for the orthographic phase

        # NLTK extension: treat -ied as a single inflectional unit
        if self.mode == self.NLTK_EXTENSIONS:
            if word.endswith("ied"):
                if len(word) == 4:
                    return self._state.replace_suffix("ied", "ie")
                else:
                    return self._state.replace_suffix("ied", "i")

        # (m>0) EED -> EE
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                # Morphological: boundary falls between stem and "ee"
                return self._state.replace_suffix("eed", "ee")
            else:
                return word

        # Find -ed or -ing
        rule_2_or_3_succeeded = False
        for suffix in ["ed", "ing"]:
            if word.endswith(suffix):
                intermediate_stem = word[: -len(suffix)]
                if self._contains_vowel(intermediate_stem):
                    rule_2_or_3_succeeded = True
                    matched_suffix = suffix
                    break

        if not rule_2_or_3_succeeded:
            return word

        # Morphological strip
        self._state.replace_suffix(matched_suffix, "")
        intermediate_stem = self._state.current

        # Everything below is orthographic -- disable recording
        self._state.recording_enabled = False
        try:
            # AT -> ATE
            if intermediate_stem.endswith("at"):
                self._state.current = intermediate_stem + "e"
                return self._state.current
            if intermediate_stem.endswith("bl"):
                self._state.current = intermediate_stem + "e"
                return self._state.current
            if intermediate_stem.endswith("iz"):
                self._state.current = intermediate_stem + "e"
                return self._state.current
            # Double-consonant deletion
            if self._ends_double_consonant(intermediate_stem) and intermediate_stem[
                -1
            ] not in ("l", "s", "z"):
                self._state.current = intermediate_stem[:-1]
                return self._state.current
            # (m=1 and *o) -> append e
            if self._measure(intermediate_stem) == 1 and self._ends_cvc(
                intermediate_stem
            ):
                self._state.current = intermediate_stem + "e"
                return self._state.current
        finally:
            self._state.recording_enabled = True

        return self._state.current

    # ------- Override _step5a / _step5b: orthographic e and ll cleanup,
    #          no boundaries should be recorded -------

    def _step5a(self, word):
        if self._state is None:
            return super()._step5a(word)
        prev = self._state.recording_enabled
        self._state.recording_enabled = False
        try:
            return super()._step5a(word)
        finally:
            self._state.recording_enabled = prev

    def _step5b(self, word):
        if self._state is None:
            return super()._step5b(word)
        prev = self._state.recording_enabled
        self._state.recording_enabled = False
        try:
            return super()._step5b(word)
        finally:
            self._state.recording_enabled = prev

    # ------- Prefix stripping -------

    def _strip_prefixes(self, word: str) -> Tuple[str, List[Tuple[int, str]]]:
        """Attempt to strip one or more productive prefixes from the start
        of `word`. Returns (remainder, [(position_in_original, prefix), ...]).

        Strategy: greedy longest-prefix match with strict validity checks
        at each step (blacklist, minimum remainder length, must contain a
        vowel). At most two prefixes are stripped to avoid over-segmentation
        (e.g. "unreliable" must not become un + re + liable).
        """
        if not self.enable_prefix_segmentation:
            return word, []

        remaining = word
        offset = 0
        stripped: List[Tuple[int, str]] = []
        # Cap at 2 prefixes; 3 or more is almost always over-segmentation
        max_depth = 2

        for _ in range(max_depth):
            matched = None
            for prefix in PRODUCTIVE_PREFIXES:
                if not remaining.startswith(prefix):
                    continue
                rest = remaining[len(prefix) :]
                if len(rest) < self.min_remainder_length:
                    continue
                if not any(c in "aeiouy" for c in rest):
                    continue
                if (prefix, rest) in PREFIX_BLACKLIST:
                    continue
                matched = prefix
                break
            if matched is None:
                break
            stripped.append((offset + len(matched), matched))
            offset += len(matched)
            remaining = remaining[len(matched) :]

        return remaining, stripped

    # ------- Main entry point -------

    def segment(self, word: str, to_lowercase: bool = True) -> List[str]:
        """Return the full morphological segmentation of `word` as an ordered
        list of surface morphemes. Prefixes come first, then the root and
        any suffixes in left-to-right order.
        """
        if to_lowercase:
            word = word.lower()

        if len(word) <= 2:
            return [word]

        # NLTK_EXTENSIONS exception pool: map the whole word to a single
        # unsegmentable morpheme
        if self.mode == self.NLTK_EXTENSIONS and word in self.pool:
            return [self.pool[word]]

        # 1. Prefix stripping
        body, prefix_strips = self._strip_prefixes(word)

        # 2. Run the modified Porter algorithm on the body
        state = _SegState(body)
        self._state = state
        try:
            # Reuse the full parent stem() pipeline; our overridden methods
            # intercept the calls and record boundaries in the state object
            _ = self._run_porter_steps(body)
        finally:
            self._state = None

        # 3. Assemble result: prefix pieces + pieces sliced from the body
        body_pieces = state.final_pieces()

        # Prepend prefix pieces if any were stripped
        if not prefix_strips:
            return body_pieces

        result = []
        prev = 0
        for pos, pref in prefix_strips:
            result.append(pref)
        # body_pieces are already in order; just append them
        result.extend(body_pieces)
        return result

    def _run_porter_steps(self, word: str) -> str:
        """Execute all PorterStemmer steps in order (mirrors the body of the
        parent's stem(), omitting the lowercase conversion and pool lookup
        that are already handled in segment()).
        """
        stem = word
        if len(word) <= 2:
            return stem
        stem = self._step1a(stem)
        stem = self._step1b(stem)
        stem = self._step1c(stem)
        stem = self._step2(stem)
        stem = self._step3(stem)
        stem = self._step4(stem)
        stem = self._step5a(stem)
        stem = self._step5b(stem)
        return stem


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seg = PorterSegmenter()

    test_cases = [
        # suffix only
        ("played", ["play", "ed"]),
        ("playing", ["play", "ing"]),
        ("running", ["run", "ing"]),  # double-n deletion
        ("flies", ["fli", "es"]),
        ("happiness", ["happi", "ness"]),
        ("nationalism", ["nation", "al", "ism"]),
        ("nationalize", ["nation", "al", "ize"]),
        ("conditional", ["condit", "ion", "al"]),
        ("hopeful", ["hope", "ful"]),
        ("dogs", ["dog", "s"]),
        ("walked", ["walk", "ed"]),
        # prefix only
        ("unhappy", ["un", "happy"]),  # "happy" should stay -- step1c not triggered
        ("redo", ["re", "do"]),
        ("disable", ["dis", "able"]),
        ("misread", ["mis", "read"]),
        # prefix + suffix
        ("unhappiness", ["un", "happi", "ness"]),
        ("unreliable", ["un", "reli", "able"]),
        ("disorganization", ["dis", "organ", "iz", "ation"]),
        ("preconditioning", ["pre", "condit", "ion", "ing"]),
        ("misunderstanding", ["mis", "under", "stand", "ing"]),
        # negative cases: prefix must NOT be stripped
        ("remember", ["rememb", "er"]),  # "re" should not be stripped
        ("input", ["input"]),  # "in" should not be stripped
        ("react", ["react"]),  # blacklisted
        # short words and exceptions
        ("sky", ["sky"]),
        ("skies", ["sky"]),
        ("a", ["a"]),
        ("the", ["the"]),
    ]

    print(f"{'word':<25} {'segmentation':<35} {'expected'}")
    print("-" * 90)
    n_match = 0
    for word, expected in test_cases:
        result = seg.segment(word)
        match = "✓" if result == expected else "✗"
        if result == expected:
            n_match += 1
        print(f"{word:<25} {' + '.join(result):<35} {' + '.join(expected)} {match}")
    print(f"\n{n_match}/{len(test_cases)} match")
