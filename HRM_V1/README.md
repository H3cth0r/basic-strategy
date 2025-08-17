# Hierarchical Reasoning Model

Paper starts defining what its reason:

It defines it as a **process of devising and executing  goal-oriented action sequences**.

Current state of LLM employ chain of though for decomposition fo tasks, which requires extensive data requirements and processing.


The papare inspired in the **hierarchical and multi-timescale processing of human brain**.

This is a recurrent architecture. **It atains significant computational depth**. 

Execute sequential reasoning tasks in a single forward pass, **without explicit supervision of intermediate process**.

Has two recurrent modules:
- high-level module responsible for **slow, abstract planning**
- Low-level module handling **rapid, detailed computations**.


With only 27 million parameters and 1000 training samples, it get exceptional performance on complex reasoning tasks.

Model operated without pre-training or CoT(Chaing Of Thought) data.







HRM is inspired in hierarchical processing and temporal separation in the brain. Two recurrent networks operating at different timescales, to collaboratively solve tasks:
- In the brain its called **Cross Frequency Coupling**:
```
Meta representation (θ wave at 4-8Hz) -> Lower level representation Y waves as 40Hz
```

- In HRM
```
High Level(slower - less timescales) <--> Low Level (faster - more timescales)
```





Deep learning = stacking more layers to achieve increased representation power and improve performance.
Fixed depth of standard transformer places them in computational complexity classes such as AC or TC,
**preventing them from solving problems that require polynomial time**.


**LLMs are not Turing-Complete and thus cannot, exercute complex algorithmic reasoning** that is necessary for **planning** or symbolic manipulation tasks.

Their results that increasing model depth can improve perfomance.


CoT externalizezd reasoning into **token-level** language by breaking down complex tasks into simpler intermediate steps, generating text. 
**CoT relies on human-defined decompositions where a single misstep or a misorder of steps can derail reasoning process entirely**.

This dependencie on explicit linguistic steps tethers reasoning to patterns at the **token-level**, which requires a lot of traning data and generates large amount of tokens for complex reasoning tasks. 



The team explore **latent reasoning*: conduct computations withing internal hidden state space. 
Which aligns with understanding that language is a tool for human communication, not the substrate of thought itself -> **brain sustains lenghtly, coherent chains fo reasoning with remarkable efficiency in latent space**, not need of constant translation back to language.




**Power of Latent Reasoning is constrained by models effective computational depth**.

Problems:
- Naively, stacking layers is notoriously generates vanishing gradients. 
- Recurrent architectures often suffer from early convergence, computational expensive and memory intensive Backpropagation.



Human brain organizes computation hierarchically across cortical regions operating at different timescales, enabling deep, multi-stage reasoning. 
Recurrent feedback loops iteratively refine internal representations, allowing slow, higer-level areas to guide and fast lower level circuits to execute. 

NOTE: what is **prohibitive credit- assignment costs that typically hamper recurrent networks from backpropagation through time**?



Two coupled recurrent modules:
- High level module for abstract deliberate reasoning
- Low-level module for fast, detailed computations.

This architecture avoids rapid convergence of standard recurrent models, through a process named : **hierarchical convergence**.
The slow updating of the High Level module advances only after the fast updfating Low level module has completed multiple compuitational steps and reached a local equilibrium, at which the low level module is reset to beging a new computational phase.

The team proposed a a **one-step gradient aproximation** for training HRM, for improved efficiency and eliminating requirement for BPTT.








## Hierarchical Reasoning Model

Inspired by three principles of neural computation **observed in the brain**:
- Hierarchical processing
- Temporal Separation
- Recurrent Connectivity



Hierarchical processing
High level areas integrated information over longern timescales and form abstract representations.
Lower level areas handle inmediate, detailed sensory and motor processing.


Temporal Separation
The hierarchical levels in the brain operate at distinct instrinsic timescales, reflected in neural rhuthms. 
The separation allos for stable, high level guidance of rapid, low-level computational.


Recurrent Connectivity
Brain features extensive recurrent connections. These feedback loops enable iterative refinement, yielding more accurate and context-sensitive
representation at the cost of additional processing time. 


NOTE: What is **Deep Credit assignment problem associated with BPTT**.




HRM consist of four learnable components:
- Input network
- Low level Recurrent module
- High level recurrent module
- Output network




Hierachical Convergence
Although convergen is crucial for recurrent networks, standard RNNs are limited by their tendency to ceonverge too early. As the hiden state settles toward a fixed poiint, update magnitudes shrink, effectively stalling subsequent computation and capping the networks effectrive depth.

To preserve computational prover, we actually want convergence to proceed very slowly-but engineering that graducal approach is difficult, since pushing convergence to far edges sysyem toward instability.

HRM designed to conteract premature convergence through **hierarchical convergence**. During each cycle, the low level module(a RNN) exhibits stable convergence to a local equilibrium. The equilibrium, however, depends on the high level state, supplied during that cycle. After completing the T steps, High level module incorpoerate the sub computations outcome, and performs its own update. THis uupdate stablished frech context for the low level module, essentially **restarting** its computational path and initiating a new convergence phase toward a different local equilibrium.




HRM perofm a sequence of distinct, stable, nested computations, where the high level module directs the overall problem solving strategy, while the low level module executes the intensive search of refinement required for each step.





They implemented **Deep Supvervision**, which is inspired by the principle that periodic neural oscillations regulate when learning occurs int the brain.




Adaptive Computational Time (ACT). Brain dynamically modulates the runtime of circuits according to task complexicty and potential rewards.
HRM incorporates mechanism that enables **thinking, fast and slow**.
