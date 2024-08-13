# llm.odin

Port of the [llm.c](https://github.com/karpathy/llm.c) code by Andrej Karpathy to [Odin](https://odin-lang.org/).

Implements the [GPT-2](https://github.com/openai/gpt-2) model with support for training and evaluation on CPU (float32 precision) or using Cuda (in bfloat16).

Tested on Linux with Cuda 12.6 and cuDNN 9.3.0

## Install

- Install [OpenBLAS](https://www.openblas.net) and set OPENBLAS_NUM_THREADS=n where n is number of available cores for acceleration on CPU.

- Install the [Cuda toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN library](https://developer.nvidia.com/cudnn)

- Install Odin as per docs at https://odin-lang.org/docs/install to clone the git repo and build the compiler

- Under root dir for this project:
  - clone this repo  `git clone git clone https://github.com/jnb666/llm.odin.git llm`
  - run tests: `cd llm/gpt2; odin test . -all-packages`
  - build exe: `cd ..; odin build . -o:speed`

## Model validation

Below commands are all run from the llm dir:

- Copy the GPT-2 124M model snapshot files: `./download_starter_pack.sh`

- Get the tiny_shakespeare dataset: `./llm prepare -dataset tiny_shakespeare`

- Run the comparison: `./llm test` or `./llm test -cuda`

## Sampling

For example to generate some text using the 124M GPT2 pre-trained model downloaded above:

`./llm generate -prompt "Large language models will often make stuff up"`

## Training

To finetine the GPT-2 124M model on the tiny shakespeare dataset run:

`./llm train -dataset tiny_shakespeare -steps 50 -val-every 5`

and to generate text from the saved checkpoint:

`./llm generate -model gpt2_124M_tiny_shakespeare.bin -nonstop -maxlen 512"`

Run `./llm <command> --help` or see the source for all the command line options.

## License
MIT