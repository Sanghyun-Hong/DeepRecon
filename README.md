## [Preprint] Security Analysis of Deep Neural Networks Operating in the Presence of Cache Side-Channel Attacks

This repository includes the code for the paper </br>
[_Security Analysis of Deep Neural Networks Operating in the Presence of Cache Side-Channel Attacks_](https://arxiv.org/abs/1810.03487)

**Authors:** [Sanghyun Hong](http://sanghyun-hong.com), Michael Davinroy, [Yigitcan Kaya](http://www.cs.umd.edu/~yigitcan), Stuart Nevans Locke, Ian Rackow, Kevin Kulda, [Dana Dachman-Soled](https://user.eng.umd.edu/~danadach/), and [Tudor Dumitras](http://users.umiacs.umd.edu/~tdumitra/). </br>
**Contact:** [Sanghyun Hong](mailto:shhong@cs.umd.edu), [Michael Davinroy](mailto:michael.davinroy@gmail.com)


## About

DeepRecon is an exemplary attack that reconstructs the architecture of the victim's DNN by using the internal information extracted via Flush+Reload, a cache side-channel technique. DeepRecon observes function invocations that map directly to architecture attributes of the victim network so that the attacker can reconstruct the victim's entire network architecture from her observations.

---

## Prerequisites


### 1. Runtime Environment

 - Ubuntu 16.04
 - Python 2.7.15-rc1
 - TensorFlow 1.10.0
 - Mastik v0.0.2


### 2. Preparations

To run DeepRecon, we require two preparation steps:

 1. Compiling TensorFlow from source to extract (only) the symbol table in use
 2. Compiling the attack code with the support of the off-the-shelf Flush+Reload library (Mastik).

#### 2.1. Build TensorFlow from Source

The official instructions on building TensorFlow can be found at this [website](https://www.tensorflow.org/install/install_sources).

##### 2.1.1 Install Bazel

Bazel is Google's build system, required to build TensorFlow. Building TensorFlow usually requires an up-to-date
version of Bazel; there is a good chance that whatever your package manager provides will be outdated. There are various ways to obtain a build of Bazel (see https://bazel.build/versions/master/docs/install-ubuntu.html).

Note: we use the specific version of Bazel (**v0.16.1**). You can download and install from [here](https://github.com/bazelbuild/bazel/releases/tag/0.16.1).


##### 2.1.2 Install Python Packages

      $ pip install -r requirements.txt

##### 2.1.3 Build and Install TensorFlow w. Bazel

Run the configuration script (currently, we disable all the features) under the *tensorflow* dir.
[Note: we recommend to use Python virtual environment, to suppress the conflicts with your system.]

      $ ./configure <<< 'n'

Compile This command will build Tensorflow using optimized settings for the current machine architecture.

      $ bazel build -c dbg --strip=never //tensorflow/tools/pip_package:build_pip_package

We still need to build a Python package using the now generated build_pip_package script.

      $ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

Now there will be a package inside /tmp/tensorflow_pkg, which can be installed with pip.

      $ pip install /tmp/tensorflow_pkg/tensorflow-<version>-<architecture>.whl


#### 2.2. Compile the Attack Source

Our attack reconstructs the DNN's architecture from the extracted attributes via Flush+Reload. We first build Mastik library that implements side-channel attacks and incorporate the library into the extraction code.

##### 2.2.1 Build Mastik

      $ ./build_mastik.sh


##### 2.2.2 Build Attack Code

      $ cd attacks
      $ make

----


## Running DeepRecon

Run the inference with the model using a sample (mug.jpg).
[Note that in the real-attack, we do not need to query the victim model; this can be achieved by a user's query while we are passively monitoring the cache behaviors.]

      $ cd models
      $ python vgg16.py
      Using TensorFlow backend.
      0 iteration, press any key to perform...

      (wait until the extraction is ready.)

After that, run the DeepRecon attack code (specify the location to store the output file).</br>
**[Note: in the flush_reload.c, the thresholds and monitor parameters should be tuned in advance.]**

      $ cd attacks
      $ ./flush_reload .
      ------------ Monitor -------------
       Searching [ 0] for [_ZN10tensorflow12_GL]: : the func. offset [   4f8be5f]
       Searching [ 1] for [_ZN10tensorflow6Bias]: : the func. offset [   7a38bb6]
       Searching [ 2] for [_ZN10tensorflow9Soft]: : the func. offset [   7e99738]
       Searching [ 3] for [_ZN10tensorflow18Una]: : the func. offset [   7e4d57a]
       Searching [ 4] for [_ZN10tensorflow7Unar]: : the func. offset [   96dc50a]
       Searching [ 5] for [_ZN10tensorflow7Unar]: : the func. offset [   957f404]
       Searching [ 6] for [_ZN10tensorflow18Una]: : the func. offset [   7e4d172]
       Searching [ 7] for [_ZN10tensorflow18Una]: : the func. offset [   7ebe46c]
       Searching [ 8] for [_ZN10tensorflow18Una]: : the func. offset [   7eea696]
       Searching [ 9] for [_ZN10tensorflow18Una]: : the func. offset [   7e4b942]
       Searching [10] for [_ZN10tensorflow18Una]: : the func. offset [   7e4bd4a]
       Searching [11] for [_ZN10tensorflow14Lau]: : the func. offset [   9d72242]
       Searching [12] for [_ZN10tensorflow8MatM]: : the func. offset [   7759dfa]
       Searching [13] for [_ZN10tensorflow12Max]: : the func. offset [   97baa4e]
       Searching [14] for [_ZN10tensorflow8Bina]: : the func. offset [   898505e]
       Searching [15] for [_ZN10tensorflow12Avg]: : the func. offset [   975f21a]
       Searching [16] for [_ZN10tensorflow12Con]: : the func. offset [   5519e1a]
       Searching [17] for [_ZN10tensorflow14Lau]: : the func. offset [   9d722c2]
      ------------- Total --------------
       Monitored: [18]
      ----------------------------------
      Do analysis of collected data

The extracted data stored into the *accesses.raw.csv* file.

      ...
      3487174,17,30,hit,End Conv
      3487190,17,28,hit,End Conv
      3487206,17,30,hit,End Conv
      3487218,2,198,hit,Softmax
      ...

----

## Running Defenses

To test the effectiveness of defenses, run the decoy processes during the above attack times or run the unraveled models instead of running the off-the-shelf model available on the Internet.

#### 1 Run Decoy Processes

Run the scripts in the *defenses/decoys* along with the victim network.

  1. **tiny_convs.py**: convolutional layer only.
  2. **tiny_relus.py**: convolutional layer + ReLU activation.
  3. **tiny_merges.py**: convolutional layer + ReLU activation + skip-connections.

#### 2. Do Obfuscations Based on the Unraveling Method

Run the **unravel_resnet50.py** script in the *defenses/obfuscations* and extract the architecture attributes.


----

## Cite This Work

You are encouraged to cite our paper if you use **DeepRecon** for academic research.

```
@article{Hong19DeepRecon,
  author    = {Sanghyun Hong and
               Michael Davinroy and
               Yigitcan Kaya and
               Stuart Nevans Locke and
               Ian Rackow and
               Kevin Kulda and
               Dana Dachman{-}Soled and
               Tudor Dumitras},
  title     = {Security Analysis of Deep Neural Networks Operating in the Presence
               of Cache Side-Channel Attacks},
  journal   = {CoRR},
  volume    = {abs/1810.03487},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.03487},
  archivePrefix = {arXiv},
  eprint    = {1810.03487},
  timestamp = {Tue, 30 Oct 2018 10:49:09 +0100},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

**Fin.**
