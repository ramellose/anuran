# anuran
Toolbox for identification of non-random associations

[![Build Status](https://travis-ci.com/ramellose/anuran.svg?token=9mhqeTh13MErxyrk5zR8&branch=master)](https://travis-ci.com/ramellose/anuran)

This toolbox is intended to identify conserved or unique associations across multiple networks.
While carrying out set operations on such networks can help you find such associations,
there is a chance that the outcome of the set operation was caused by random overlap between the networks.
_anuran_ helps you identify if your biological networks have set operations that have different outcomes than would be expected by chance.
The simulated case study included with _anuran_ can also help you design your experiments. The paper describing this case study is currently in preparation.
Contact the author at lisa.rottjers (at) kuleuven.be. Your feedback is much appreciated!
This version is still in early alpha and has been tested for Python 3.6.

## Getting Started

To install _anuran_, run:
```
pip install git+https://github.com/ramellose/anuran.git
```


For a quick demo, run _anuran_ as follows, with the output filepath changed to something suitable for your system.
When specifying the filepath, include the full path and a prefix for naming files.
For example, _anuran_ will save a csv file to anuran_demo_sets.csv with the command below.
```
anuran -i demo -o C://Documents//anuran_demo -draw -sample 10
```

The demo data for _anuran_ was downloaded from the following publication:
Meyer, K. M., Memiaghe, H., Korte, L., Kenfack, D., Alonso, A., & Bohannan, B. J. (2018).
Why do microbes exhibit weak biogeographic patterns?. The ISME journal, 12(6), 1404.

To run the script, only two arguments are required: input and output filepaths.
The script recognizes gml, graphml and txt files by their extension.
The text files should be edge lists, with the third column containing edge weight.
```
anuran -i filepath1.graphml filepath2.gml filepath3.txt -o filepath_to_output
```

_anuran_ generates null models with permutations of the original network.
By default, two models are generated: one that changes the degree distribution
and one that does not.
Note that the model changing the degree distribution may not have a major effect
on the network structure as most smaller networks will not have enough dyad pairs to swap, especially if degree assortativity is large.

```
anuran -n deg        # preserves degree
anuran -n random     # changes degree
```

The set sizes are calculated for the difference set and the intersection set by default.
You can also choose to calculate the set sizes for all possible intersections of a number of the networks, i.e. each edge that was present on 2 out of 5 networks.
If you flag the sign option, signs of edge weights are taken into account.
For this, the input networks need to have weight as an edge property.
The set difference can then have edges that have a unique edge sign in one network but a different edge sign in all others.
In contrast, the set intersection will only include edges that have the same sign across the networks.

```
anuran -set difference intersection       # Default calculates sizes of difference and intersection sets
anuran -size 0.2 0.4 0.6                        # Calculates null models for edges present in partial intersections
anuran -sign                              # Includes edge sign in set calculation
```

If you want to know how the set sizes change when you increase the number of replicates,
use the parameter below; this will calculate set sizes for all numbers of networks up to the total.
```
anuran -sample 10
```

For a complete explanation of all the parameters, run:
```
anuran -h
```

For documentation of specific functions, check out [the Sphinx documentation](https://ramellose.github.io/anuran/index.html).

### Contributions

This software is still in early alpha. Any feedback or bug reports will be much appreciated!

## Authors

* **Lisa Röttjers** - [ramellose](https://github.com/ramellose)
* **Karoline Faust** - [hallucigenia-sparsa](https://github.com/hallucigenia-sparsa)

See also the list of [contributors](https://github.com/ramellose/anuran/contributors) who participated in this project.

## License

This project is licensed under the Apache License - see the [LICENSE.txt](LICENSE.txt) file for details


