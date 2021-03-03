so, I'm not going to get to this tonight, but I'm thinking that Henry, Hynek, Thomas are right about removing upper bounds but it also feels like we're abandoning users to figure it out themselves if we remove all of the compatible release synatx that we have in our extras at the moment.

I'm going to write up a Gist to formalize my thoughts and try to have some discussion on why NumPy and SciPy don't follow the recommendations of Hynek.
But I'm thinking of the idea of adding a module to contrib that can generate a "recommendation" `requirements.txt` based on the release and extras a user wants.
So something like

```shell
pyhf contrib recommended --extras xmlio > requirements.txt
```

which would generate something like

```shell
$ cat requirements.txt
# core dependencies
scipy~=1.4
click~=7.0
tqdm~=4.56
jsonschema~=3.2
jsonpatch~=1.23
pyyaml~=5.1
# xmlio
uproot3~=3.14
uproot~=4.0
```
