## Note
- in gpu machine in one minute we proceed with 3 step with batch size of 32.
- out of memory issue: [link](https://discuss.pytorch.org/t/cpu-ram-usage-increases-inside-each-epoch-and-keeps-increasing-for-all-epochs-oserror-errno-12-cannot-allocate-memory/89682/5)
storing losses without detacthing them causes accumulation of data.

## TODO
- in gan class couldn't fully understand how lossed for discrimator are calculated. Its not same us original GAN.
