<<机器学习>>的课程项目要求


课程的Project的要求，可以从如下几个方面开展工作：

1.maBandits,已经给出了stochastic Bandit相关的所有代码。第一个版本是matlab的软件。
包括了，UCB，KLUCB，policyKLUCB,  exp3，policyThompson等经典的MAB算法。（20分）

这个代码的在线的随机数列，是用bernulli分布，高斯分布等产生的随机过程，来模拟数据集合的arm和对应的每个arm收益的。

请从Main函数入手，代码结构是很清晰的。

2. EXP3++,给出了一些EXP3，还有UCB相关的代码。第二个版本是matlab的软件。EXP3.pdf是这个代码对应的理论论文（20分）


这个代码的在线的随机数列，是用指数分布产生的随机过程，来模拟数据集合的arm和对应的每个arm收益的。

请从Main函数入手，代码结构是很清晰的。

3. 论文《 Follow the Leader If You Can, Hedge If You Must》对应的AdaHedge, 也是一个bandit的进阶版的代码。有些批注是德语办的，作者已经翻译了很多成为英语了。如果有些comment不懂，可以借助google translate翻译成英语。（25分）
给出了Hedge， FTL，FlipFop，FTL，Hazan-Kale等在学习速率变化等情况下，regret的performance

请从adahedge.m函数入手，代码结构是很清晰的。

4. 不用上述code。自己去找其他的bandit的code，比如matlab和python语言的，在github上面。只要你能搞懂代码的工作原理，画出regret的曲线和其他相关曲线，替换变成实际的数据集合，也是可以的，并且有加分。（30分）

5. 实际数据集（dataset）：可以包含各种各样你再互联网上你找到的机器学习相关的数据集合dataset，可以包括但不限定：信息安全、区块链、医疗健康、航空与航天、机器人robutics、社交网络social netwokrks、外卖点餐takeout order与评分and ratings数据集合、车联网Internet of Vehicles、无人机unmanned aerial vehicles (UAVs)、自动驾驶selfdriving、智能交通Intelligent Transportation、计算机视觉，多媒体信号处理Multimedia and signal processing、智能电网、MOOC慕课、控制科学、物联网Internet of Things(IoT)与物联网安全IoT security、工业物联网industrial Internet of Things(IoT)、无线通信、信息网络、网络安全network security、生物与细胞结构大数据、医药与疾病数据、制造业与CAD大数据、农业大数据、船舶与海洋大数据、图形大数据、等等。在google，github等网站，按照这些关键词去收索。

6.具体要求如下

请读懂这些代码，替换掉他们用标准的分布函数做成的随机序列，替换为实际的数据集合从新run一些实际数据集的数据，提交课程报告，即可。实际数据集合，包含的应该是含有{数据，标签（reward）}的序列。


报告的内容，请用word的4号字体写一个大概8-10页左右的报告。包含，你对这些代码的理解。算法的原理描述，加入自己的理解，有些算法的描述，可以直接从上面的算法对应的论文中 photo copy，但是一定要加入中文的解释，体现你的理解。比如，EXP3算法的每一个参数是什么意思。

每个regret的曲线要对比至少4个不同的算法，比如KLUCB，policyKLUCBexp，policyThompson。这些都没有在课堂上讲过，但是互联网上都有相关的论文，请自己下载一下自己读读。花不了太多时间的。这些算法的代码我都给出来了。

代码的结构和怎么实现这个代码的过程，要描述清楚。得到的regret曲线，不同的算法的好坏程度。要比较，给出好了大概多少得百分比，比如KL-UCB，在XXX数据集上面比UCB算法在给定多少run的情况下，regret的大小少了30%之类的话。

你找到的实际数据集合，对应在这个实际数据集合，要保存，实际数据集，你再网上找到的link也要写在帮助文档里面。连同你的修改后的matlab和其他你找到的代码，以及你run的代码得到的regret learning的曲线图，打一个包，放在一起。压缩成一个包。
命名为M***学号_姓名.zip的格式。

项目报告用word完成。命名为M***学号_姓名.doc的格式。

同学们可以一起讨论这些代码，搞懂这个run这些代码。但是报告不能抄袭，要按照自己的理解来写，数据集必须每个人的来源都不同。如果违反了这个规定，就按照抄袭论处了。

这个成绩平均占总成绩的20%。如果你的作业完成的做得不够好，做项目选最难的那个，这个比例可以提高到35%，可以作为成绩的加分项。请务必重视。
