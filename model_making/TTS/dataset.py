from typing import List


def text_to_sequence(text: str, cleaner_names) -> List[int]:
    """ Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    pass


def sequence_to_text(sequence: List[int]) -> str:
    """Converts a sequence of IDs back to a string"""
    pass
