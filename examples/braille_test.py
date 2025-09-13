import numpy as np
import matthewplotlib.core as mpc

dots = np.array([
    [0,1,0,1,1,0,1,1,0,1,0,0,1,1,0,1,1,0,1,1,0,1,1,0],
    [1,1,0,0,1,0,1,1,0,1,1,0,1,0,0,1,1,0,1,1,0,1,1,0],
    [1,1,0,1,1,0,0,1,0,1,1,0,1,1,0,1,0,0,1,1,0,1,1,0],
    [1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,1,0,1,0,0],
    [1,0,0,1,0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0],
    [1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
    [1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
    [1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
    [1,1,1,1,0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
    [1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
    [1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
    [1,0,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,0],
]).astype(bool)
print("dot patterns")
print(dots.astype(int))

braille_codepoints = mpc.braille_encode(dots)
print("braille codepoints")
print(braille_codepoints)

print("braille characters")
for row in braille_codepoints:
    braille_characters = [chr(int(b)) for b in row]
    print("".join(braille_characters))
