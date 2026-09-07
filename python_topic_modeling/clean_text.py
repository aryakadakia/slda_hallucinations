"""Repair transcript encoding damage.

UTF-8 punctuation in these transcripts had its high bits stripped at some
point in the chain, so a right single quote (U+2019 = E2 80 99) survives as
the three bytes 62 00 19, which reads as 'b' followed by a control character.
The visible effect is that every contraction is mangled: it's -> itb\x19s.

That matters because tokenisation then produces junk terms ("itbs", "donbt")
instead of real words, which is why those strings appear in the project's
hand-written stopword list.
"""
import re

# high-bit-stripped UTF-8 punctuation -> the character originally intended
REPAIRS = {
    '\x19': "'",   # U+2019 right single quote
    '\x18': "'",   # U+2018 left single quote
    '\x1c': '"',   # U+201C left double quote
    '\x1d': '"',   # U+201D right double quote
    '\x14': ' - ', # U+2014 em dash
    '\x13': ' - ', # U+2013 en dash
    '\x26': '...', # U+2026 ellipsis
}
_PAT = re.compile('b([' + ''.join(REPAIRS) + '])')

def repair(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = _PAT.sub(lambda m: REPAIRS[m.group(1)], text)
    # any surviving stray control characters
    out = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', out)
    # cp1252/utf-8 double encoding, in ~8% of the baseline file. Some of it is
    # mis-decoded emoji (the "Å¸" fragment is the giveaway), which the round trip
    # cannot restore to anything meaningful.
    if re.search(r'Ã|â€|Â|Å', out):
        try:
            out = out.encode('cp1252', 'ignore').decode('utf-8', 'ignore')
        except Exception:
            pass
    # Whatever survives is residue, not language. Drop the whole affected WORD
    # rather than stripping the offending characters out of it: several of these
    # are mangled contractions, and surgically removing the non-ASCII turns
    # "don<junk>t" into "donut" and "it<junk>s" into "itus" - real-looking words
    # that then enter the vocabulary and reach the topics. Deleting the token
    # loses one word; repairing it invents one.
    out = ' '.join('' if any(ord(ch) > 127 for ch in w) else w for w in out.split())
    return re.sub(r'\s+', ' ', out).strip()

if __name__ == '__main__':
    demo = "Yes, I know itb\x19s been a couple of days since yb\x19all have heard from me"
    print("before:", repr(demo))
    print("after :", repr(repair(demo)))
