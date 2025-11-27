# HW4
2024010729 刘武韬
## Experiment 1
我分别在GSM8K与Math500两个数数据集上采取zeroshot，1_shot, 3_shots 与 5_shots 进行试验，最终结果如下
| Accuracy | zeroshot | 1_shot | 3_shots | 5_shots 
|-|-|-|-|-|
|GSM8K|0.846|0.834|0.838|0.87|
|Math500|0.69|0.732|0.748|0.722|

由Experiment1结果可以看出，few_shot prompt在GSM8K问题上的提升并不明显，可能是由于该数据集中数学问题较为简单，模型无需学习就能够有效解决问题；而在Math500数据集上，few_shot prompt对准确率有明显提升，shot=1与shot=3相较zero_shot正确率提升了4～5个百分点；而当shot=5时，正确率虽然相较zero_shot有提升，但是模型表现并不再随示例数量提升而变好。

以上结果说明，当few_shot prompt数量在一定范围内时会提升模型表现，zero_shot prompt会让模型在遇到相对较复杂问题时正确性下降，而few_shot prompt示例过多时则会影响模型对上下文的读取判断，输入信息可能会冗余，最终影响模型最终决策。

## Experiment 2

在Experiment1中我们可以发现，few shot prompt在gsm8k上对模型表现帮助有限，这可能因为该数据集中问题难度有限，zero shot prompt便能使模型有很好的表现，导致模型向demonstration学习所带来的收益小于或于约等于模型处理更长文本带来的注意力消耗。

为了更好的展示few shot prompt对模型表现的提升，我构造了一个自定义计算的问题，分别进行了zero_shot prompt与few_shot prompt，输入与结果如下

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

**分析：** Zero-shot的回答是**错误的**。模型最终输出"3D5"而非正确的"CDE"，说明模型没有完全理解Zeta notation的转换规则——虽然计算正确，但在转换回Zeta notation时混用了数字和字母。

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

**分析：** Few-shot的回答是正确的。通过示例，模型学会了Zeta notation的完整规则，它能够正确理解字母到数字的映射与位置规则，并且更加简洁地推导出正确答案。这说明few shot prompt在处理自定义符号系统时具有明显优势，能够通过示例教会模型新的规则。

## Experiment 3
我在Math500数据集上分别采取greedy decoding 与 majority voting的方式进行试验。其中majority voting 分别选取 N=5， N=10进行试验，每个试验的正确率如下

| Accuracy | greedy decoding | voting_5 | voting_10 | voting_20 |
|-|-|-|-|-|
|Math500|0.69|0.688|||