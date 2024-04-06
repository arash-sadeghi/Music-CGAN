## TODO
- [ ] Validation data bass is patches of rendom bass track and the output is mixed with generated drum and the orginal one. Validation data should be from
a single song which should be before random shuffling in data loader.
- [ ] it should be checked that pitches in piano roll correspond to the same ones in logivPro
- [ ] we need three datasets. One is just rock songs to evaluate DP. the other is for a fragment of continues sonf as validation data. And the other is whole dataset for calculating EB.
- in gan class couldn't fully understand how lossed for discrimator are calculated. Its not same us original GAN.
- with temp 120 each beat is 0.5 second. With 4/4 rythms which says a measure should have 4 beats, then a measure should las4 4*0.5s = 2s

## Note
- in gpu machine in one minute we proceed with 3 step with batch size of 32.
- out of memory issue: [link](https://discuss.pytorch.org/t/cpu-ram-usage-increases-inside-each-epoch-and-keeps-increasing-for-all-epochs-oserror-errno-12-cannot-allocate-memory/89682/5)
storing losses without detacthing them causes accumulation of data.
