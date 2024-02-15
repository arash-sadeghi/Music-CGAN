# problem

how should i embed a bass sample?

nn.embedding is designed to work on indexes not on vectors.

The nn. Embedding layer is a simple lookup table that maps an index value to a weight matrix of a certain dimension . This simple operation is the foundation of many advanced NLP architectures, allowing for the processing of discrete input symbols in a continuous space

solution1: interpret bass peice as a binary number. use that binary number to feed to embedding. the only problem with that is that binary number has a huge range for a matrix of size 64*72. The range is 2**(64*72). we can scale that number to sth between 0 and some reasonable number. like 1000. the problem with this is that some drum tracks might get scaled to same number because that number at the end will be integer. for now a good starting point.

solution2: check code implementation of musegan and midinet.

solution3: use transformers.

solution4: count distinctive bass measures and use their index. The problem with that is 
# changes
- we are only dealing with one track both in conditioner and generator. also piano roll has no depth because we are dealing only with one track. So we can just conv2d since there is no depth to go into.
- in conv2d we need to change kernel size since we are not interested in depth information anymore, because we dont have depth information
- generator layeras took a lot of tweaking to generate desired output size.
- for generator condition will be contcatinated in depth
- third dimention is not just for instruments. it is also for measures. this way gan considers time structures but seperately generating measures and concating them. for now I will work without time structure and only 2d matrices. but in future we can introduce 3rd dimention as measures so that gan will learn time structure
    - for now I only let discruminator to have ed conv and have understanding of measure.
    - my only concern for now is that generator and conditioner does not have 3d understanding like discriminator
- for now we are training to learn drum track at time t for bass t. but we can in future say learn drum track time t+1 for bass t so that drummer catches up with bassist.
- in gan class couldn't fully understand how lossed for discrimator are calculated. Its not same us original GAN.
