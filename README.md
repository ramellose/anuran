# anuran ![numma](https://github.com/ramellose/anuran/blob/master/anuran.png)
Toolbox for identification of non-random associations

[![Build Status](https://travis-ci.com/ramellose/anuran.svg?token=9mhqeTh13MErxyrk5zR8&branch=master)](https://travis-ci.com/ramellose/anuran)
[![HitCount](http://hits.dwyl.com/ramellose/anuran.svg)](http://hits.dwyl.com/ramellose/anuran)

This toolbox is intended to identify conserved or unique patterns across multiple networks.
While carrying out set operations on such networks can help you find such associations,
there is a chance that the outcome of the set operation was caused by random overlap between the networks.
_anuran_ helps you identify if your biological networks have set operations that have different outcomes than would be expected by chance.
The simulated case study included with _anuran_ can also help you design your experiments. The paper describing this case study is currently in preparation.
Contact the author at lisa.rottjers (at) kuleuven.be. Your feedback is much appreciated!
This version is still in development and has been tested for Python 3.6.

## Getting Started

First set up a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/).
```
virtualenv venv
# Linux
source venv/bin/activate

# Windows
venv/Scripts/activate

# Once you are done with anuran:
deactivate
```

To install _anuran_, run:
```
pip install git+https://github.com/ramellose/anuran.git
```

If you have Python 2.7 installed as your default Python, please ensure you are installing and running anuran on your Python 3 version.
```
python3 -m pip install git+https://github.com/ramellose/anuran.git
```
For a quick demo, run _anuran_ as follows, with the output filepath changed to something suitable for your system.
When specifying the filepath, include the full path and a prefix for naming files.
For example, _anuran_ will save a csv file to anuran_demo_sets.csv with the command below.

```
anuran -i demo -o C://Documents//anuran_demo -draw -perm 10 -nperm 10
```

The demo data for _anuran_ was downloaded from the following publication: <br />
Meyer, K. M., Memiaghe, H., Korte, L., Kenfack, D., Alonso, A., & Bohannan, B. J. (2018).
Why do microbes exhibit weak biogeographic patterns?. The ISME journal, 12(6), 1404. <br />
For a more elaborate demo analysis, please check out [the vignette](https://ramellose.github.io/anuran/demo_anuran.html).

To run the script, only two arguments are required: input and output filepaths.
The script recognizes gml, graphml and txt files (without headers) by their extension.
The text files should be edge lists, with the third column containing edge weight.
```
anuran -i folder_with_networks -o filepath_to_output
```

You can specify more than one folder with the above parameter. If you want to compare across folders in addition to the
null model comparison, add the flag below.
```
anuran -compare
```

_anuran_ can also calculate gradients for ordered networks.
If you have ordered networks, for example by constructing networks along a spatial or temporal gradient, _anuran_ will test whether
there is a correlation in network properties compared to the null models.
To use this feature, you need to use a naming convention for your networks.
Within a folder, order the networks with a number and underscore, like below.
```
1_networkname.extension
2_networkname.extension
```

_anuran_ generates null models with permutations of the original network.
By default, two models are generated: one that changes the degree distribution
and one that does not.
Note that the model changing the degree distribution may not have a major effect
on the network structure as most smaller networks will not have enough dyad pairs to swap, especially if degree assortativity is large.
You can specify the number of null models with the parameter below.
```
anuran -perm              # number of randomized networks
```

It is possible to generate randomized networks with a specific core.
Since we do not know the true core, multiple randomizations (equal to the total number of networks) are generated from each input network,
and these are then used in comparisons.

If you want to simulate networks where 30% or 50% of the associations is shared across half of the networks,
you can add the parameters below. Note that you can fill in more than one value.
```
anuran -cs 0.3 0.5        # core size
anuran -prev 1            # core prevalence
```

The set sizes are calculated for the difference set and the intersection set by default.
You can also choose to calculate the set sizes for all possible intersections of a number of the networks, i.e. each edge that was present on 2 out of 5 networks.
If you flag the sign option, signs of edge weights are taken into account.
For this, the input networks need to have weight as an edge property.
The set difference can then have edges that have a unique edge sign in one network but a different edge sign in all others.
In contrast, the set intersection will only include edges that have the same sign across the networks.
```
anuran -set difference intersection       # Default calculates sizes of difference and intersection sets
anuran -size 0.2 0.4 0.6                  # Calculates null models for edges present in partial intersections
anuran -sign                              # Includes edge sign in set calculation
```

If you want to know how the set sizes change when you increase the number of replicates,
use the parameter below; this will calculate set sizes for all numbers of networks up to the total number of networks. Up to 10 combinations are considered here.
```
anuran -sample 10
```

The sample size calculation can quickly become slow. If you are not interested in observing every single value, you can specify the sample numbers you are interested in.
```
anuran -n 5 10 20
```

In addition to set sizes, you can compute centrality scores (degree, betweenness and closeness centrality) and graph properties (assortativity, average shortest path length, connectivity, diameter and radius).
If you want to know whether the set sizes, centralities or graph properties are different from the null models,
you can run some statistics on these values. Note that the statistics are not reliable if you have fewer than 20 networks!

The centrality scores and graph properties are compared across networks using a [Mann-Whitney _U_-test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test).
The Mann-Whitney _U_-test does not require equal _n_; this is important since not all networks contain the same nodes. If there are too many unique nodes across networks,
this test may give strange results. Hence. if you want to carry out these statistical tests, make sure that prevalence of most nodes is high across networks.
For comparing graph properties, the p-value is computed using [the standard score](https://en.wikipedia.org/wiki/Standard_score). This assumes that graph properties are normally distributed.

To compute centralities and graph properties and the associated test (with Bonferroni multiple-testing correction), add the following parameters:
```
anuran -c -net -stats bonferroni
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


