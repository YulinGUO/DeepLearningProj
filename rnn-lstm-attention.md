## 从RNN说起
 Recurrent neural network,RNN的主要用途是处理和预测序列数据。Fully connected NN或者CNN中，网络结构都是从输入层到隐含层再到输出层，层与层之间是全连接或者部分连接，但是每层之间的节点是无连接的。如果要预测橘子的下一个单词是什么，一般需要当前单词以及前面的单词，因为句子中前后单词并不是独立的。RNN会记忆之前的信息，并利用之前的信息影响后面节点的输出。也就是说，RNN的Hidden Layer之间的节点是有连接的，隐藏层的输入不仅仅包含Input Layer的输出，也包括上一时刻的Hidden Layer的输出。
  ![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png)
  上图为一个典型的RNN。在每个时刻t，RNN会针对该时刻的输入结合当前模型的状态给出一个输出，并更新模型状态。RNN的主体结构A除了来自Input Layer $x_t$ ,还有个循环的提供上一时刻的隐藏状态 hidden state : $h_{t-1}$。在每一个时刻，rnn的模块A在读取了$x_t$以及$h_{t-1}$之后会生成新的隐藏状态$h_t$,并产生本时刻的输出$o_t$,有时候$h_t$也可以视为输出。由于模块A中的运算和变量在不同时刻是相同的，因此rnn理论上可以看作成同意神经网络被无限复制的结果。
  将完整的输入输出序列展开，可以得到下图所示的结果： 
  ![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)
  RNN对长度为N的序列展开后，可以视为一个有N个中间层的前馈神经网络。这个前馈神经网络没有循环连接，因为可以使用Back Propagation 进行训练。这种方法称为：Back Propagation Through Time。
  RNN中的状态是通过一个向量来表示的，这个向量的维度也称为RNN Hidden Layer的大小。
  ![](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png)
  上图所示的RNN，表示一个使用单层，并且使用tanh作为激活函数的全连接神经网络作为循环体。 
  
##LSTM
在有些问题中，模型仅仅需要短期内的信息来执行当前的任务，并不需要记忆这个短语之前更长的上下文信息。但遇到一些上下文场景更加复杂的情况，仅仅根据短期依赖无法很好的解决这种问题，因为当前预测位置和相关信息之间的文本间隔有可能很大，而简单的RNN就可能会丧失学习到距离如此远的能力。而Long Short-term memory, LSTM的设计就是为了解决这个问题。与单一的tanh循环体结构不同，LSTM是一种拥有三个gate的网络结构：
![](https://www.yunaitong.cn/media/14585445702525/14585500063294.png)
LSTM靠一些门的结构让信息有选择的影响神经网络中每个时刻的状态。所谓门，就是一个使用sigmoid神经网络和一个按位做乘法的操作，这两个操作合在一起就是一个门结构。Sigmoid层输出0~1之间的值，每个值表示对应的部分信息是否应该通过。0值表示不允许信息通过，1值表示让所有信息通过。一个LSTM有3个这种门，来保护和控制状态 $C$。
LSTM的第一步是决定我们将要从状态中扔掉哪些信息。该决定由一个叫做“遗忘门(Forget Gate)”的Sigmoid层控制。遗忘门观察$h_{t−1}$和$x_t$，对于状态$C_{t−1}$中的每一个元素，输出一个0~1之间的数。1表示“完全保留该信息”，0表示“完全丢弃该信息”。
![](https://www.yunaitong.cn/media/14585445702525/14585517843913.jpg)
下一步是决定我们将会把哪些新信息存储到状态C中。这步分为两部分。首先，有一个叫做“输入门(Input Gate)”的Sigmoid层决定我们要更新哪些信息。接下来，一个tanh层创造了一个新的候选值，$Ct$~，该值可能被加入到状态中。在下一步中，我们将会把这两个值组合起来用于更新状态。
![](https://www.yunaitong.cn/media/14585445702525/14585522130400.jpg)
我们把旧状态$C_{t−1}$乘以$f_t$，忘掉我们已经决定忘记的内容。然后我们再加上$i_t∗C_t$~.
![](https://www.yunaitong.cn/media/14585445702525/14585647039038.jpg)
最后，我们需要决定最终的输出。输出将会基于目前的状态，并且会加入一些过滤。首先我们建立一个Sigmoid层的输出门(Output Gate)，来决定我们将输出状态C的哪些部分。然后我们将状态通过tanh之后（使得输出值在-1到1之间），与输出门相乘，这样我们只会输出我们想输出的部分。
![](https://www.yunaitong.cn/media/14585445702525/14585652046323.jpg)

## Seq2Seq
Seq2Seq模型的基本思想是使用一个rnn来读取输入句子，将整个句子的信息压缩到一个固定维度的编码中，再使用另一个rnn读取这个编码，将其解压为目标语言的一个句子。这两个rnn分别称为编码器Encoder,解码器Decoder，这个模型也称为Encoder-decoder模型。
![](https://pic4.zhimg.com/80/v2-b2f4e56107dc06e4916a70d899e46203_hd.jpg)
解码器的结构与语言模型几乎相同：输入为单词的词向量，输出为softmax层产生的单词概率，损失函数为log perplexity。 事实上，解码器可以理解为一个以输入编码为前提的语言模型(Conditional Language Model)。语言模型中使用的共享softmax层和词向量的参数，都可以直接用到Seq2Seq中。
编码器部分则更为简单，它与解码器一样拥有词向量层以及rnn，但是由于编码阶段无输出，因此不需要softmax层。
在训练过程中，编码器顺序读入每个单词的词向量，然后将最终的hidden state复制到解码器作为初始状态。解码器的第一个输入是<sos>Start of Sentence字符。每一步预测的单词是训练数据的目标句子，预测序列的最后一个单词是与语言模型相同的<eos>End of sentence.

普通Seq2Seq不足:
1.信息的有损压缩。虽然Decoder接收到了输入序列的信息，但它却不能掌握完全的信息。因为我们的Encoder是对输入序列进行了有损压缩。那就意味着在传递信息的过程中，对信息有一定的损失。而且如果我们输入的句子越长，那么在压缩过程中，损失的信息量就越大，进而导致Decoder对于模型的预测结果就越不好。
![](https://pic4.zhimg.com/80/v2-5b4510552a695e5b479f735ae5f370a8_hd.jpg)
上面这个图说明了这个问题，随着句子长度的增加，BLEU会先增加后减小。也就意味着，对于一般句子的长度来说（10-20词之间），模型的翻译效果是相当好的，但是当句子越来越长，它的结果只会变得越来越差。
2.RNN的时间维度过大。当句子序列很长时，这就意味着RNN的时间维度很深。例如，当一个句子长度为50的时候，这就意味着在RNN训练过程中，它需要递归50次来进行计算。那么就会导致在BPTT过程中出现梯度弥散的现象。即使我们使用像LSTM这种模型来解决这个问题，但梯度弥散仍然是一个要考虑到的问题。
3.Context Vector的同质性。Encoder将输入序列转化为Context Vector以后传递给Decoder，Decoder在进行每一个词进行翻译的时候，它所参考的输入序列的信息都是Context Vector这样一个常量。而在实际场景中，我们在对一个句子进行翻译的时候，不会去过多关注与我们当前翻译词不相关的词。例如，翻译”I love machine learning“的时候，当我们翻译到”machine learning“的时候，我们实际上是不会关注前面是”I love“还是”He loves“，我们只关心”machine learning“这个词组的翻译为”机器学习“。所以在使用相同的Context Vector翻译时，会包含一些干扰信息，使得结果不够准确。

##Attention机制
在Seq2Seq模型中，编码器将完整的输入句子压缩到一个维度固定的向量中，然后解码器根据这个向量生成输出句子。当输入句子较长，这个中间向量难以储存足够的信息，Attention机制就是为了解决这个难题而设计，Attention机制允许解码器随时查阅输入句子中的部分单词或片段，因此不需要中间向量保存所有信息。
首先我们先定义以下变量：
![](https://pic2.zhimg.com/80/v2-c5e2e784df15cbf2f4170a6c7fdbcf8c_hd.jpg)
在原来的Encoder-Decoder模型中，我们实际上在最大化联合概率分布：
![](https://pic3.zhimg.com/80/v2-60697366c09700b58d5260ae160e5fd8_hd.jpg)
其中，每一个条件概率可以表示为：
![](https://pic4.zhimg.com/80/v2-15c1d04431413390fc6e5abf9b612253_hd.jpg)
其中g是非线性函数，也就是我们的RNN。

在Attention模型中，我们的的每一个条件概率变为：
![](https://pic1.zhimg.com/80/v2-ca8e2785133ebf468d77d2c8ab6fb509_hd.jpg)
其中第i个阶段状态是关于前一个阶段状态、前一阶段输出以及第i个Context Vector的函数。
![](https://pic4.zhimg.com/80/v2-b4cc91508480c86c691c40d2acba2dc4_hd.jpg)
从新的模型中，我们可以发现，在对第i个目标值进行预测时，概率函数开始利用整个X输入的信息，意味着对于Decoder端，在翻译每一个单词的时候，都会有独一无二的Context Vector与它对应，这就解决了Context Vector同质性的问题。
那么这里的Context Vector该如何计算呢？它实际上是一个对Encoder端每个阶段状态h的加权结果：
![](https://pic4.zhimg.com/80/v2-2bb70cd9b1fdfbf4774cb9749ae1a7b5_hd.jpg)
其中权重α的定义如下：
![](https://pic4.zhimg.com/80/v2-4b7c11a77c14905fb7236467a3fc7f72_hd.jpg)
这里的权重计算方式有点类似softmax的计算。其中e被称为alignment model，它评价了输入语句中第j个单词与输出语句中第i个单词的匹配程度。在不同的论文中，这些e的计算方式也不同。无论采用哪个模型，通过softmax计算权重$\alpha$和通过权重计算context的方法都是一样的。

 ![](https://pic1.zhimg.com/80/v2-3d45ea51a64c62840e891fb1aae72251_hd.jpg)
 上图展示了我们在预测第t个阶段的输出y时的结构。通过对Encoder层状态的加权，从而掌握输入语句中的所有细节信息。
 以机器翻译为例（将中文翻译成英文）：
 ![](https://pic1.zhimg.com/80/v2-d266bf48a1d77e7e4db607978574c9fc_hd.jpg)
 输入的序列是“我爱中国”，因此，Encoder中的h1、h2、h3、h4就可以分别看做是“我”、“爱”、“中”、“国”所代表的信息。在翻译成英语时，第一个上下文c1应该和“我”这个字最相关，因此对应的 $a_{11}$ 就比较大，而相应的 $a_{12}$ 、 $a_{13}$ 、 $a_{14}$ 就比较小。c2应该和“爱”最相关，因此对应的 $a_{22}$ 就比较大。最后的c3和h3、h4最相关，因此 $a_{33}$ 、 $a_{34}$ 的值就比较大。
 
## 参考
TensorFlow，实战Google深度学习模型
https://zhuanlan.zhihu.com/p/27608348
https://zhuanlan.zhihu.com/p/28054589
 




