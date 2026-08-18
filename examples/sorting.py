"""
Animated sorting algorithms using columns.

Visualises multiple sorting algorithms simultaneously using `mp.columns`. The height
of each column represents the value, and colors are used to highlight elements
currently being compared or swapped. Algorithms race side by side.

By Gemini 3.1 Pro.
"""

import tyro
import time
import numpy as np
from typing import Literal

import matthewplotlib as mp

Algo = Literal["bubble", "insertion", "selection", "gnome", "quick", "merge"]


def bubble_sort(arr):
    arr = list(arr)
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            yield arr, [j, j+1]
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                yield arr, [j, j+1]
                swapped = True
        if not swapped:
            break
    yield arr, []


def insertion_sort(arr):
    arr = list(arr)
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        yield arr, [i, j]
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
            yield arr, [j + 1, j] if j >= 0 else [0]
        arr[j + 1] = key
        yield arr, [j + 1]
    yield arr, []


def selection_sort(arr):
    arr = list(arr)
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            yield arr, [min_idx, j]
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        yield arr, [i, min_idx]
    yield arr, []


def gnome_sort(arr):
    arr = list(arr)
    n = len(arr)
    index = 0
    while index < n:
        if index == 0:
            index += 1
        yield arr, [index, index - 1]
        if arr[index] >= arr[index - 1]:
            index += 1
        else:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            index -= 1
            yield arr, [index, index + 1]
    yield arr, []


def quick_sort(arr):
    arr = list(arr)
    def _quick_sort(low, high):
        if low < high:
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                yield arr, [j, high]
                if arr[j] <= pivot:
                    i = i + 1
                    arr[i], arr[j] = arr[j], arr[i]
                    yield arr, [i, j]
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            yield arr, [i + 1, high]
            pi = i + 1
            
            yield from _quick_sort(low, pi - 1)
            yield from _quick_sort(pi + 1, high)
            
    yield from _quick_sort(0, len(arr) - 1)
    yield arr, []


def merge_sort(arr):
    arr = list(arr)
    def _merge_sort(l, r):
        if l < r:
            m = l + (r - l) // 2
            yield from _merge_sort(l, m)
            yield from _merge_sort(m + 1, r)
            
            left = arr[l:m+1]
            right = arr[m+1:r+1]
            i = j = 0
            k = l
            while i < len(left) and j < len(right):
                yield arr, [k, min(m+1+j, r)]
                if left[i] <= right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                yield arr, [k]
                k += 1
            while i < len(left):
                arr[k] = left[i]
                i += 1
                yield arr, [k]
                k += 1
            while j < len(right):
                arr[k] = right[j]
                j += 1
                yield arr, [k]
                k += 1
                
    yield from _merge_sort(0, len(arr) - 1)
    yield arr, []


def get_generator(algo: Algo, arr):
    if algo == "bubble": return bubble_sort(arr)
    if algo == "insertion": return insertion_sort(arr)
    if algo == "selection": return selection_sort(arr)
    if algo == "gnome": return gnome_sort(arr)
    if algo == "quick": return quick_sort(arr)
    if algo == "merge": return merge_sort(arr)
    raise ValueError(f"Unknown algorithm: {algo}")


def main(
    n: int = 30,
    fps: int = 30,
    algos: tuple[Algo, ...] = ("selection", "quick", "merge", "insertion"),
    save: str | None = None,
    num_frames: int = 0,
):
    """Animate sorting algorithms in parallel."""
    # Generate random data so all algos sort the same array
    np.random.seed(42)
    values = np.random.rand(n) * 100
    
    generators = [get_generator(algo, values) for algo in algos]
    states = [next(g) for g in generators]
    done = [False] * len(algos)
    
    animation = mp.animate(
        fps=fps,
        record=save is not None,
        stop_on_interrupt=True,
    )
    
    with animation as anim:
        frame = 0
        while not all(done):
            plots = []
            for idx, g in enumerate(generators):
                if not done[idx]:
                    try:
                        states[idx] = next(g)
                    except StopIteration:
                        done[idx] = True
                
                arr, active = states[idx]
                
                if done[idx]:
                    # Win state!
                    colors = [(144, 238, 144)] * n
                else:
                    # Active sorting state
                    colors = ["red" if i in active else "white" for i in range(n)]
                
                p = mp.columns(
                    arr,
                    height=15,
                    column_width=1,
                    column_spacing=0,
                    vrange=(0, 100),
                    colors=colors,
                )
                
                p = mp.border(
                    p,
                    title=f" {algos[idx].title()} Sort ",
                    style=mp.BoxStyle.HEAVY,
                )
                plots.append(p)
            
            # Combine all plots dynamically wrapping to terminal width
            anim.update(mp.wrap(*plots))
            
            frame += 1
            if num_frames > 0 and frame >= num_frames:
                break

    if save:
        anim.frames.savegif(save, bgcolor="black")


if __name__ == "__main__":
    tyro.cli(main)
