## TODO
- [ ] Validation data bass is patches of rendom bass track and the output is mixed with generated drum and the orginal one. Validation data should be from
a single song which should be before random shuffling in data loader.
- [ ] it should be checked that pitches in piano roll correspond to the same ones in logivPro
- [ ] we need three datasets. One is just rock songs to evaluate DP. the other is for a fragment of continues sonf as validation data. And the other is whole dataset for calculating EB.

## info
- in gan class couldn't fully understand how lossed for discrimator are calculated. Its not same us original GAN.
- with temp 120 each beat is 0.5 second. With 4/4 rythms which says a measure should have 4 beats, then a measure should las4 4*0.5s = 2s
