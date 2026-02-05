# HW4
2024010729 刘武韬

## Experiment 1

本实验在GSM8K和Math500两个数据集上分别采用zero-shot、1-shot、3-shot和5-shot进行对比实验，实验结果如下表所示：

| 数据集 | Zero-shot | 1-shot | 3-shot | 5-shot |
|--------|-----------|--------|--------|--------|
| GSM8K  | 0.846     | 0.834  | 0.838  | 0.870  |
| Math500| 0.690     | 0.732  | 0.748  | 0.722  |

从实验结果可以看出，few-shot prompt在GSM8K数据集上的提升并不明显，这可能是因为该数据集中的数学问题相对简单，模型在zero-shot设置下已经能够有效解决问题；而在Math500数据集上，few-shot prompting对准确率有明显提升，其中1-shot和3-shot相比zero-shot正确率提升了约4-5个百分点。然而，当shot数量增加到5时，虽然准确率相比zero-shot仍有提升，但模型表现不再随示例数量增加而持续改善。

以上结果表明，few-shot prompt的示例数量存在一个最优范围。当示例数量过少时，模型在面对相对复杂的问题时正确率会下降；而当示例数量过多时，可能会影响模型对上下文的读取和判断，输入信息冗余可能导致模型决策能力下降。

## Experiment 2

从Experiment 1的结果可以发现，few-shot prompt在GSM8K数据集上对模型表现的提升有限，这可能是因为该数据集中的问题难度相对较低，zero-shot prompting已经能使模型达到较好的表现，导致从demonstration中学习所带来的收益小于或等于处理更长文本所带来的注意力消耗。

为了更好地展示few-shot prompt对模型表现的提升作用，我设计了一个自定义的符号系统计算问题，分别进行了zero-shot和few-shot实验，输入与结果如下：

### Zero-Shot
```
Question: In Zeta notation: A=1, B=2, C=3, D=4, E=5. What is CAD + CA? Express your answer in Zeta notation.
Answer:
```
**Response:**

To solve the expression $CAD + CA$ using the given Zeta notation values where $A = 1$, $B = 2$, $C = 3$, $D = 4$, and so on, we first need to convert each term into its numerical value.

1. **Convert $CAD$ to a numerical value:**
   - $C = 3$, $A = 1$, $D = 4$
   - Therefore, $CAD = 3 \times 10^2 + 1 \times 10^1 + 4 \times 10^0 = 300 + 10 + 4 = 314$.

2. **Convert $CA$ to a numerical value:**
   - $C = 3$, $A = 1$
   - Therefore, $CA = 3 \times 10^1 + 1 \times 10^0 = 30 + 1 = 31$.

3. **Add the two numerical values together:**
   - $314 + 31 = 345$.

4. **Convert the sum back into Zeta notation:**
   - $345$ can be broken down as $3 \times 10^2 + 4 \times 10^1 + 5 \times 10^0$.
   - Using the Zeta notation values, this translates to $3D5$.

Therefore, the answer in Zeta notation is $3D5$.

**分析：** Zero-shot的回答是**错误的**。模型最终输出"3D5"而非正确的"CDE"，说明模型没有完全理解Zeta notation的转换规则。虽然数值计算部分是正确的（CAD=314, CA=31, 314+31=345），但在将结果转换回Zeta notation时，模型混用了数字和字母，未能正确应用字母到数字的映射规则。

### Few-Shot
```

Question: In Zeta notation: A=1, B=2, C=3, D=4, E=5. What is AB + ACC? Express your answer in Zeta notation.
Answer: AB = 12, ACC = 133. AB + ACC = 12 + 133 = 145 = ADE. The answer is ADE."

Question: In Zeta notation: A=1, B=2, C=3, D=4, E=5. What is BAD + AA? Express your answer in Zeta notation.
Answer: BAD = 214, AA = 11. BAD + AA = 214 + 11 = 225 = BBE. The answer is BBE."

Question: In Zeta notation: A=1, B=2, C=3, D=4, E=5. What is CAD + CA? Express your answer in Zeta notation.
Answer:
```

**Response:**

CAD = 314, CA = 31. CAD + CA = 314 + 31 = 345 = CDE. The answer is CDE.

**分析：** Few-shot的回答是**正确的**。通过提供的两个示例，模型学会了Zeta notation的完整转换规则，能够正确理解字母到数字的映射关系以及位置规则，并且能够简洁地推导出正确答案"CDE"。这说明few-shot prompt在处理自定义符号系统或非标准任务时具有明显优势，能够通过示例有效地教会模型新的规则和格式要求。

## Experiment 3
**Please Note**: 由于分配api_key的问题，以下实验均使用阿里云平台提供的API接口调用qwen2.5-7b-instruct模型，模型准确率普遍偏低，Experiment1与2的结果对该实验无参考价值。


本实验在Math500数据集上分别采用greedy decoding（baseline，相当于N=1）和majority voting策略进行对比。其中majority voting分别设置N=5和N=10，温度参数T=0.8。实验结果如下表所示：

| 数据集 | Greedy Decoding | Majority Voting (N=5) | Majority Voting (N=10) |
|--------|-----------------|----------------------|------------------------|
| Math500| 0.602           | 0.662                | 0.680                  |

实验结果表明，majority voting策略相比greedy decoding在准确率上有显著提升。当N=5时，准确率从0.602提升到0.662，提升了约6.0个百分点；当N=10时，准确率为0.680，相比baseline提升了约7.8个百分点。值得注意的是，N=10时的准确率高于N=5，这表明随着采样数量增加，majority voting能够通过更多的样本覆盖更广泛的答案空间，从而获得更可靠的投票结果。这说明在温度采样设置下，增加采样数量有助于提升最终答案的准确性。




## Experiment 4

Math500数据集中的题目按照难度分为1-5五个等级，等级越高表示难度越大。本实验按照Experiment 3中的不同采样策略，分组统计了各个难度等级下的准确率，结果如下表所示：

| 难度等级 | 题目数量 | Greedy Decoding | Majority Voting (N=5) | Majority Voting (N=10) |
|----------|----------|-----------------|----------------------|------------------------|
| Level 1  | 43       | 0.791           | 0.837                | 0.814                  |
| Level 2  | 90       | 0.711           | 0.744                | 0.744                  |
| Level 3  | 105      | 0.695           | 0.743                | 0.771                  |
| Level 4  | 128      | 0.602           | 0.648                | 0.711                  |
| Level 5  | 134      | 0.396           | 0.500                | 0.493                  |

从分层分析结果可以看出，majority voting策略在不同难度等级上的提升效果存在差异，且N=5和N=10的表现也有所不同。

对于较简单的题目（Level 1-3），两种majority voting策略都能带来准确率提升。Level 1在N=5时达到0.837（提升4.6个百分点），在N=10时为0.814（提升2.3个百分点）；Level 2在N=5和N=10时均为0.744（提升3.3个百分点）；Level 3在N=5时为0.743（提升4.8个百分点），在N=10时达到0.771（提升7.6个百分点）。

对于较难的题目（Level 4-5），majority voting的提升效果更为明显。Level 4在N=5时为0.648（提升4.6个百分点），在N=10时达到0.711（提升10.9个百分点）；Level 5在N=5时为0.500（提升10.4个百分点），在N=10时为0.493（提升9.7个百分点）。

值得注意的是，在Level 1和Level 4上，N=5的表现优于N=10；而在Level 3上，N=10的表现明显优于N=5。这表明不同难度等级的题目对采样数量的敏感度不同。总体而言，majority voting策略通过多次采样和投票机制，能够有效减少单次推理的错误，从而显著提升整体准确率，特别是在处理复杂问题时效果更为明显。

## Bonus 1

本实验在Math500数据集上使用不同参数量级的Qwen2.5系列模型（3B、7B、14B）分别采用zero-shot、1-shot、3-shot和5-shot进行对比实验，以探究模型规模对整体性能和few-shot prompting敏感性的影响。实验结果如下表所示：

| 模型参数量 | Zero-shot | 1-shot | 3-shot | 5-shot |
|-----------|-----------|--------|--------|--------|
| qwen2.5-3b | 0.548     | 0.622  | 0.618  | 0.634  |
| qwen2.5-7b | 0.690     | 0.732  | 0.748  | 0.722  |
| qwen2.5-14b| 0.740     | 0.770  | 0.780  | 0.771  |

从实验结果可以看出，模型规模对整体性能和few-shot prompting的敏感性都有显著影响。

随着模型参数量的增加，模型在Math500数据集上的整体准确率呈现明显的提升趋势。在zero-shot设置下，3B模型的准确率为0.548，7B模型提升至0.690（提升约14.2个百分点），14B模型进一步提升至0.740（相比3B提升约19.2个百分点）。这表明更大的模型参数量能够带来更强的数学推理能力，即使在没有任何示例的情况下也能取得更好的表现。

不同规模的模型对few-shot prompt的响应存在差异。对于3B模型，few-shot prompt带来的提升相对有限（1-shot相比zero-shot提升约7.4个百分点，3-shot和5-shot的提升幅度相近），且准确率在1-shot后基本趋于稳定。对于7B模型，few-shot prompt的效果更为明显（1-shot相比zero-shot提升约4.2个百分点，3-shot达到峰值0.748），但5-shot时准确率略有下降（0.722），这可能是因为示例数量过多导致模型注意力分散。对于14B模型，few-shot prompt的提升幅度相对较小（1-shot相比zero-shot提升约3.0个百分点），且3-shot和5-shot的准确率相近（0.780和0.771），说明更大规模的模型在zero-shot设置下已经具备较强的推理能力，few-shot prompting的边际收益递减。

总体而言，模型规模的增加能够显著提升整体性能，但同时也降低了模型对few-shot prompting的敏感性。较小规模的模型（如3B）更依赖few-shot示例来提升性能，而大规模模型（如14B）在zero-shot设置下已经能够取得较好的表现，few-shot prompting的额外收益相对有限。