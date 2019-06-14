# pile

Abelian Sandpiles

Instructions in the comments.

You'll need Tensorflow 2. I installed it with the command

  `pip install tensorflow==2.0.0-beta0`
  
On OSX it'll run slow as TF doesn't support GPUs on OSX.
It does use multiple cores though, if you have them.

On Linux with an nvidia GPU it should be fast.

I think you can run TF on Windows.

Note: if the `doublings` parameter is 0, then the algorithm is a straight run and the intermediate files produced can be interpreted as the story from beginning to end. If `doublings` is > 0 then it's a slightly faster algorithm but the intermediates aren't so nicely ordered. So set `doublings` to zero (and read the accompanying comments) if you want to make a video of the evolution of a sandpile.

![Example](ex1.png)
