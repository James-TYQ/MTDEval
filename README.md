<div align="center">

<h1 align="center"> Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple Judges  </h1>

<div align=center><img src="figure/MTDEval.png" width="90%" height="100%" /></div>

<p></p>
</div>

**MTDEval** is a lightweight open-source evaluator that can efficiently and flexibly evaluate multi-turn dialogues for both single rating and pairwise comparison tasks. Unlike traditional evaluators that suffer from various biases or incur significant computational overhead during inference, MTDEval captures the collective wisdom of multiple LLM judges by aggregating their preference knowledge into a single model. By leveraging a large-scale, multi-judge annotated preference dataset and employing a learning-to-rank framework, it provides reliable, interpretable, and fine-grained quality assessments. MTDEval’s robustness and applicability make it a valuable tool for advancing the evaluation and optimization of LLMs in diverse real-world dialogue scenarios.

<h3 id="3.1">🔧 Installation</h3>

First clone the repository:
```bash
git clone https://github.com/James-TYQ/MTDEval
```

Next, set up a conda environment:
```bash
conda create -n MTDEval python=3.10.9
conda activate MTDEval
```
Then, install the dependencies using pip:
```bash
pip install -r requirements.txt
```

<h3 id="3.2">⏩ Quickly Start </h3>
<!-- 如需采用 MTDEval 的数据格式进行训练，请首先按照 /data/P^2-MTD 文件夹中的数据格式构建训练集。需要注意的是，我们支持 two distinct models in our experiments: one for overall rating，在overall文件夹下； and another for evaluating the performance across ten specific dimensions, 在Multi_Dim文件夹下. 可使用如下命令进行模型训练 -->

If you need to train using MTDEvalr's format, first construct the training data according to the data format in `data/train/seeds.json`. Then, use the following command to train:

```bash
bash train_Multi.sh   # Our setting achieves good performance, but if you want to train a better result, you should adjust some parameters
```

<h3 id="3.3">📜 Tips </h3>

1. The train data examples lie in the `data/train/seeds.json` file.
2. The MD-Eval Benchmark is available at `data/benchmark/MD-Eval`.
3. The OOD data we used to evaluate the dimension selection performance of SaMer and baselines lies in `data/benchmark/OOD`.

