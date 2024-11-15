# pile

Abelian Sandpiles
-----------------

The project contains four small applications to simulate abelian sandpiles.
Each one has its own directory.

![Example](xxx.28.png)

Acorn Atom
----------
In `Atom/`.
This is a version in 6502 assembly language for the Acorn Atom.
Simply load it up into an Acorn Atom (or emulator) and run.

Swift/Metal
-----------
In `Metal/`

Rewritten for Swift/Metal.
Extremely unpolished. Read the source code in `main.swift`. Change parameters by modifying the source :)
Build with `make`.
Run with `./pile <n>` and it'll do `2^n` grains dropped in the centre.
It just renders the bottom right corner.
Use `stitch.sh <in> <out>` to construct the full image by symmetry.
(I'm too lazy to study ImageMagick. So it doubles the middle row and column.)

Jax version
-----------
In `Jax/`.
Ported to jax. Much faster than TF version.
I think it's still slower than raw CUDA.

Install jax with `pip install jax jaxlib`.
Install PIL with `pip install pillow`.

Run with `python jaxpile.py`.

TensorFlow version
------------------
In `TensorFlow/`.
Options and instructions in the comments.

You'll need Tensorflow 2. I installed it with the command

  `pip install tensorflow==2.0.0-beta1`
or  
  `pip install tensorflow-gpu==2.0.0-beta1`
  
On OSX it'll run slow as TF doesn't support GPUs on OSX.
It does use multiple cores though, if you have them.
It should output messages like
```
    Wrote to 'out.0000.0000.jpg'
    Wrote to 'out.0000.0001.jpg'
```
These are checkpoints in the process. The last image is the final reduced sandpile.

On Linux with an nvidia GPU it should be faster.
Still way slower than hand constructed CUDA code.

I think you can run TF on Windows.

Note: if the `doublings` parameter is 0, then the algorithm is a straight run and the intermediate files produced can be interpreted as the story from beginning to end. If `doublings` is > 0 then it's a slightly faster algorithm but the intermediates aren't so nicely ordered. So set `doublings` to zero (and read the accompanying comments) if you want to make a video of the evolution of a sandpile.
