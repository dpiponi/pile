# pile

Abelian Sandpiles
-----------------

Rewritten for Swift/Metal.
Extremely unpolished. Read the source code in `main.swift`. Change parameters by modifying the source :)
Build with `make`.
Run with `./pile <n>` and it'll do `2^n` grains dropped in the centre.
It just renders the bottom right corner.
Use `stitch.sh <in> <out>` to construct the full image by symmetry.
(I'm too lazy to study ImageMagick. So it doubles the middle row and column.)

Old Jax version
---------------

Ported to jax. Much faster than TF version.
I think it's still slower than raw CUDA.
Install jax with `pip install jax jaxlib`.
Install PIL with `pip install pillow`.
Run with `python jaxpile.py`.
Docs much the same as old docs for jax.py.

Old TF version
--------------
Instructions in the comments.

You'll need Tensorflow 2. I installed it with the command

  `pip install tensorflow==2.0.0-beta0`
or  
  `pip install tensorflow-gpu==2.0.0-beta0`
  
On OSX it'll run slow as TF doesn't support GPUs on OSX.
It does use multiple cores though, if you have them.

On Linux with an nvidia GPU it should be faster.
Still way slower than hand constructed CUDA code. I'll release that
when I've cleaned it up.

I think you can run TF on Windows.

Note: if the `doublings` parameter is 0, then the algorithm is a straight run and the intermediate files produced can be interpreted as the story from beginning to end. If `doublings` is > 0 then it's a slightly faster algorithm but the intermediates aren't so nicely ordered. So set `doublings` to zero (and read the accompanying comments) if you want to make a video of the evolution of a sandpile.

![Example](ex1.png)
