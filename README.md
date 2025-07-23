<div align="center">

<h1 align="center"> Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple Judges  </h1>

<div align=center><img src="figure/MTDEval.png" width="90%" height="100%" /></div>

<p></p>
</div>

**MTDEval** is a lightweight open-source evaluator that can efficiently and flexibly evaluate multi-turn dialogues for both single rating and pairwise comparison tasks. Unlike traditional evaluators that suffer from various biases or incur significant computational overhead during inference, MTDEval captures the collective wisdom of multiple LLM judges by aggregating their preference knowledge into a single model. By leveraging a large-scale, multi-judge annotated preference dataset and employing a learning-to-rank framework, it provides reliable, interpretable, and fine-grained quality assessments. MTDEval’s robustness and applicability make it a valuable tool for advancing the evaluation and optimization of LLMs in diverse real-world dialogue scenarios.

<h3 id="3.1">⬇️ Step 1: Installation</h3>

To train or inference MTDEval, first clone the repository;

Next, set up a conda environment to manage the dependencies:
```bash
conda create -n MTDEval python=3.10.9
conda activate MTDEval
```
Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

<h3 id="3.2">🚀 Quickly Start </h3>
```

If you need to train using SaMer's format, first construct the training data according to the data format in `data/train/seeds.json`. Then, use the following command to train:

```bash
bash train_Multi.sh   # Our setting achieves good performance, but if you want to train a better result, you should adjust some parameters
```

<h3 id="3.3">📜 Tips </h3>

1. The train data examples lie in the `data/train/seeds.json` file.
2. The MD-Eval Benchmark is available at `data/benchmark/MD-Eval`.
3. The OOD data we used to evaluate the dimension selection performance of SaMer and baselines lies in `data/benchmark/OOD`.
5. `SaMer-llama3-8B` checkpoint is available [here](https://drive.google.com/file/d/1jyZg-SfLVSjWE4G7sic-3VN_g62qK51l/view?usp=sharing).

